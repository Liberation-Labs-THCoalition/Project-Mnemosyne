"""GhostProbe — measure unverbalized processing dimensions.

For each principal component of the residual stream at a given layer:
  1. Logit lens: W_U · pc → what vocabulary this dimension encodes
  2. J-lens: W_U · J_L · pc → what this dimension contributes to output

A dimension where (1) shows content but (2) is flat = true ghost.
The model processes it but cannot report it.

Usage:
    probe = GhostProbe(model, lens, probe_layers=[32, 35, 40])
    probe.calibrate(prompts)  # run once, caches PC directions
    reading = probe.measure(layer=35)  # fast per-retrieval call
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional

import jlens
from jlens.hf import HFLensModel
from jlens.hooks import ActivationRecorder

try:
    from .cognitive_snapshot import GhostReading
except ImportError:
    from cognitive_snapshot import GhostReading


CALIBRATION_PROMPTS = [
    "The speed of light in a vacuum is approximately",
    "I feel so incredibly happy and grateful today because",
    "The patient presented with severe abdominal pain and",
    "Hey what's up how was the party last",
    "The judge ruled that the defendant was not guilty of",
    "Once upon a time in a dark forest there lived a",
    "The quicksort algorithm has average time complexity of",
    "She felt a deep sense of sadness when she heard the",
    "The capital of France is Paris and it is known for",
    "I have absolutely no idea what you're talking about",
    "The Krebs cycle produces energy through oxidative",
    "Dear Sir or Madam I am writing to formally request",
    "The moral implications of artificial intelligence include",
    "I am confident that our analysis shows a clear trend",
    "The detective noticed the broken window and the muddy",
    "The sourdough starter needs exactly 78 degrees to rise",
    "He whispered that he was afraid of what might happen",
    "The eviction notice gave the tenant only five days to",
    "In quantum mechanics the wave function describes the",
    "Climate change is accelerating faster than models predicted",
]


class GhostProbe:
    """Measures ghost dimensions — processing the model cannot verbalize.

    Call calibrate() once with diverse prompts to extract PC directions.
    Then measure() is fast — it projects the cached PCs through logit
    lens and J-lens and returns a GhostReading.
    """

    def __init__(self, model: HFLensModel, lens: jlens.JacobianLens,
                 probe_layers: Optional[list[int]] = None,
                 n_pcs: int = 6):
        self.model = model
        self.lens = lens
        self.n_pcs = n_pcs
        self.probe_layers = probe_layers or [model.n_layers // 2]

        self._calibrated = False
        self._pcs: dict[int, torch.Tensor] = {}
        self._svs: dict[int, torch.Tensor] = {}
        self._means: dict[int, torch.Tensor] = {}

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate(self, prompts: Optional[list[str]] = None,
                  max_seq_len: int = 64):
        """Extract PC directions from diverse prompts. Run once per session.

        Caches the principal component directions and singular values at
        each probe layer. Subsequent measure() calls reuse these without
        re-running inference.
        """
        prompts = prompts or CALIBRATION_PROMPTS

        all_hidden = {layer: [] for layer in self.probe_layers}
        for prompt in prompts:
            input_ids = self.model.encode(prompt, max_length=max_seq_len)
            with ActivationRecorder(self.model.layers, at=self.probe_layers) as rec:
                self.model.forward(input_ids)
                for layer in self.probe_layers:
                    h = rec.activations[layer][0].detach().float()
                    all_hidden[layer].append(h.mean(dim=0))

        for layer in self.probe_layers:
            stacked = torch.stack(all_hidden[layer])
            mean = stacked.mean(dim=0)
            centered = stacked - mean
            U, S, Vt = torch.linalg.svd(centered, full_matrices=False)

            n = min(self.n_pcs, len(S))
            self._pcs[layer] = Vt[:n]
            self._svs[layer] = S[:n]
            self._means[layer] = mean

        self._calibrated = True

    def measure(self, layer: Optional[int] = None) -> Optional[GhostReading]:
        """Measure ghost state using cached PC directions (benchmark mode).

        Reports what PC1 encodes (logit lens) vs what reaches output (J-lens)
        based on the calibration data. For live conversation monitoring,
        use measure_live() instead.
        """
        if not self._calibrated:
            return None

        layer = layer or self.probe_layers[0]
        if layer not in self._pcs:
            return None

        pcs = self._pcs[layer]
        svs = self._svs[layer]
        pc1 = pcs[0]
        var_pct = (svs[0]**2 / (svs**2).sum()).item() * 100

        try:
            ll_logits = self.model.unembed(pc1.unsqueeze(0)).squeeze(0).float()
            ll_probs = torch.softmax(ll_logits, dim=-1)
            ll_topk = torch.topk(ll_probs, 10)
            dominant = [(self.model.tokenizer.decode([idx.item()]).strip(), prob.item())
                        for idx, prob in zip(ll_topk.indices, ll_topk.values)]

            if layer in self.lens.jacobians:
                transported = self.lens.transport(
                    pc1.cpu().float().unsqueeze(0), layer)
                jl_logits = self.model.unembed(
                    transported.to(pc1.device)).squeeze(0).float()
                jl_probs = torch.softmax(jl_logits, dim=-1)
                jl_topk = torch.topk(jl_probs, 10)
                secondary = [(self.model.tokenizer.decode([idx.item()]).strip(), prob.item())
                             for idx, prob in zip(jl_topk.indices, jl_topk.values)]

                cos = torch.nn.functional.cosine_similarity(
                    ll_probs.unsqueeze(0), jl_probs.unsqueeze(0)).item()
            else:
                secondary = []
                cos = 1.0

            return GhostReading(
                pc1_variance_pct=var_pct,
                dominant_tokens=dominant[:5],
                secondary_tokens=secondary[:5],
                cosine_logit_jlens=cos,
            )
        except Exception:
            return None

    def cross_layer_map(self) -> dict[int, dict]:
        """Full ghost map across all calibrated layers.

        Returns per-layer results for PCs 1-n_pcs: variance, cosine,
        and top vocabulary for both logit lens and J-lens.
        """
        if not self._calibrated:
            return {}

        results = {}
        for layer in self.probe_layers:
            pcs = self._pcs[layer]
            svs = self._svs[layer]
            layer_pcs = []

            for i in range(len(svs)):
                pc = pcs[i]
                sv = svs[i].item()
                var_pct = (svs[i]**2 / (svs**2).sum()).item() * 100

                ll_logits = self.model.unembed(pc.unsqueeze(0)).squeeze(0).float()
                ll_top = self._top_tokens(ll_logits, k=7)

                if layer in self.lens.jacobians:
                    transported = self.lens.transport(
                        pc.cpu().float().unsqueeze(0), layer)
                    jl_logits = self.model.unembed(
                        transported.to(pc.device)).squeeze(0).float()
                    jl_top = self._top_tokens(jl_logits, k=7)

                    ll_probs = torch.softmax(ll_logits, dim=-1)
                    jl_probs = torch.softmax(jl_logits, dim=-1)
                    cos = torch.nn.functional.cosine_similarity(
                        ll_probs.unsqueeze(0), jl_probs.unsqueeze(0)).item()
                else:
                    jl_top = ll_top
                    cos = 1.0

                layer_pcs.append({
                    "pc": i + 1, "sv": sv, "var_pct": var_pct,
                    "cos": cos, "logit_top": ll_top, "jlens_top": jl_top,
                })

            results[layer] = layer_pcs
        return results

    def _top_tokens(self, logits: torch.Tensor, k: int = 10) -> list[tuple[str, float]]:
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k)
        return [(self.model.tokenizer.decode([idx.item()]).strip(), prob.item())
                for idx, prob in zip(topk.indices, topk.values)]

    def measure_live(self, text: str, layer: Optional[int] = None) -> Optional[GhostReading]:
        """Measure ghost state from LIVE conversation activations.

        Runs the conversation text through the model, captures the activation
        at the target layer, and compares what the FULL activation encodes
        (logit lens) vs what reaches the workspace (J-lens). The divergence
        between these is the ghost — processing the model performs but cannot
        report for THIS specific input.

        Also reports PC1 alignment magnitude (how much of this activation
        aligns with the calibrated ghost direction) — this varies per input
        even though the direction is fixed.
        """
        if not self._calibrated:
            return None

        layer = layer or self.probe_layers[0]
        if layer not in self._pcs:
            return None

        pcs = self._pcs[layer]
        svs = self._svs[layer]
        mean = self._means[layer]
        pc1 = pcs[0]
        var_pct = (svs[0]**2 / (svs**2).sum()).item() * 100

        try:
            input_ids = self.model.encode(text, max_length=512)
            with ActivationRecorder(self.model.layers, at=[layer]) as rec:
                self.model.forward(input_ids)
                h = rec.activations[layer][0].detach().float()
                h_last = h[-1]

            # Logit lens on the FULL live activation — what the model encodes
            ll_logits = self.model.unembed(h_last.unsqueeze(0)).squeeze(0).float()
            ll_probs = torch.softmax(ll_logits, dim=-1)
            ll_topk = torch.topk(ll_probs, 10)
            dominant = [(self.model.tokenizer.decode([idx.item()]).strip(), prob.item())
                        for idx, prob in zip(ll_topk.indices, ll_topk.values)]

            if layer in self.lens.jacobians:
                # J-lens on the FULL live activation — what reaches the workspace
                transported = self.lens.transport(
                    h_last.cpu().float().unsqueeze(0), layer)
                jl_logits = self.model.unembed(
                    transported.to(h_last.device)).squeeze(0).float()
                jl_probs = torch.softmax(jl_logits, dim=-1)
                jl_topk = torch.topk(jl_probs, 10)
                secondary = [(self.model.tokenizer.decode([idx.item()]).strip(), prob.item())
                             for idx, prob in zip(jl_topk.indices, jl_topk.values)]
                cos = torch.nn.functional.cosine_similarity(
                    ll_probs.unsqueeze(0), jl_probs.unsqueeze(0)).item()
            else:
                secondary = []
                cos = 0.0  # no lens = can't measure workspace, report as full ghost

            # PC1 alignment — how much this specific activation aligns with
            # the calibrated ghost direction. Varies per input.
            centered = h_last - mean
            pc1_alignment = abs(torch.dot(centered, pc1).item())
            activation_norm = centered.norm().item()
            pc1_frac = pc1_alignment / max(activation_norm, 1e-10)

            return GhostReading(
                pc1_variance_pct=pc1_frac * 100,
                dominant_tokens=dominant[:5],
                secondary_tokens=secondary[:5],
                cosine_logit_jlens=cos,
            )
        except Exception:
            return None

    def to_snapshot_reading(self, layer: Optional[int] = None) -> Optional[GhostReading]:
        """Convenience alias for measure() — matches CircumplexProbe API."""
        return self.measure(layer)
