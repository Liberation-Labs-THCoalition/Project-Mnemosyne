"""MnemosyneIntegration — hooks metacognitive measurement into the retrieval pipeline.

This is the glue between Mnemosyne's SIRA retrieval and the measurement
probes (workspace, circumplex, ghost). On each retrieval event, it:
1. Runs the workspace probe (what's in J-space?)
2. Runs the circumplex probe (what emotional geometry is active?)
3. Reads the ghost state (what's in the shadow?)
4. Records a CognitiveSnapshot
5. Returns the snapshot alongside the retrieval result

The agent's memory system continues to work normally — this layer
is observational, not interventional. It watches cognition without
changing it.
"""

import hashlib
import time
from typing import Optional

import torch
import jlens
from jlens.hf import HFLensModel

from cognitive_snapshot import (
    CognitiveSnapshot, JSpaceReading, GhostReading,
    MemoryLoadingResult, CognitiveMemoryStore,
)
from workspace_probe import WorkspaceProbe, MemoryProbe
from circumplex_probe import CircumplexProbe


class MetacognitiveObserver:
    """Observes and records the cognitive state during retrieval events.

    Attach to a Mnemosyne instance to enable metacognitive memory.
    Does not modify retrieval behavior — purely observational.
    """

    def __init__(self, model: HFLensModel, lens: jlens.JacobianLens,
                 store_path: str, agent_id: str,
                 workspace_layers: Optional[list[int]] = None,
                 circumplex_layer: Optional[int] = None):
        self.model = model
        self.lens = lens
        self.agent_id = agent_id

        self.workspace_probe = WorkspaceProbe(model, lens)
        self.circumplex_probe = CircumplexProbe(model, lens)
        self.store = CognitiveMemoryStore(store_path, agent_id)

        self.workspace_layers = workspace_layers or [35, 39, 43, 45, 47]
        self.circumplex_layer = circumplex_layer or 45
        self.model_name = ""

    def observe_retrieval(self, memory_id: str, memory_content: str,
                          task_prompt: str, retrieval_method: str = "sira",
                          significance: float = 0.5,
                          session_id: str = "",
                          marker_tokens: Optional[list[str]] = None) -> CognitiveSnapshot:
        """Record a full cognitive snapshot at a retrieval event.

        Call this after SIRA (or any retriever) returns a memory,
        before the memory is used in generation.
        """
        timestamp = time.time()

        # 1. Workspace readings at key layers
        ws_readings = self._measure_workspace(memory_content, task_prompt)

        # 2. Circumplex at the ignition layer
        circ = self._measure_circumplex()

        # 3. Ghost state
        ghost = self._measure_ghost()

        # 4. Memory loading verification
        loading = None
        if marker_tokens:
            loading = self._measure_loading(
                memory_id, memory_content, task_prompt, marker_tokens)

        # 5. Assemble snapshot
        snapshot = CognitiveSnapshot(
            timestamp=timestamp,
            session_id=session_id,
            agent_id=self.agent_id,
            memory_id=memory_id,
            memory_content_hash=hashlib.sha256(memory_content.encode()).hexdigest()[:16],
            retrieval_method=retrieval_method,
            significance_score=significance,
            workspace_readings=ws_readings,
            workspace_onset_layer=self._find_onset(ws_readings),
            dominant_workspace_tokens=self._dominant_tokens(ws_readings),
            circumplex=circ,
            ghost=ghost,
            loading=loading,
            model_name=self.model_name,
            n_layers=self.model.n_layers,
            d_model=self.model.d_model,
            lens_prompts=self.lens.n_prompts,
        )

        # 6. Record
        self.store.record(snapshot)

        return snapshot

    def _measure_workspace(self, context: str, task: str) -> list[JSpaceReading]:
        """J-lens readings at workspace layers."""
        from jlens.vis import compute_slice

        prompt = f"Context:\n- {context}\n\nQuestion: {task}\nAnswer:"
        slice_data = compute_slice(
            self.model, self.lens, prompt,
            top_n=10, max_seq_len=512,
        )

        readings = []
        n_pos = slice_data.seq_len
        last_pos = max(0, n_pos - 1)

        for layer_idx, layer_num in enumerate(slice_data.layers):
            if layer_num not in self.workspace_layers:
                continue

            top_at_pos = slice_data.top_ids[last_pos, layer_idx, :]
            vocab = slice_data.vocab_fragment

            tokens = []
            for rank, tid in enumerate(top_at_pos[:10]):
                tid = int(tid)
                tok_str = vocab.get(tid, f"<{tid}>")
                tokens.append((tok_str, 1.0 / (rank + 1)))

            # Approximate cos by checking if top tokens align between
            # logit lens (raw unembed) and J-lens (transported unembed)
            cos = 0.0  # Would need full computation for accuracy
            rand = 0.0

            readings.append(JSpaceReading(
                layer=layer_num,
                top_tokens=tokens,
                cosine_logit_jlens=cos,
                random_baseline=rand,
                in_workspace=True,  # These ARE workspace layers
            ))

        return readings

    def _measure_circumplex(self) -> Optional:
        """Circumplex reading at the ignition layer."""
        try:
            result = self.circumplex_probe.measure_at_layer(self.circumplex_layer)
            return self.circumplex_probe.to_snapshot_reading(result)
        except Exception:
            return None

    def _measure_ghost(self) -> Optional[GhostReading]:
        """Ghost dimension state at mid-network."""
        from jlens.hooks import ActivationRecorder

        mid_layer = self.model.n_layers // 2
        # Use a neutral prompt to read the ghost state
        input_ids = self.model.encode("The ", max_length=4)

        try:
            with ActivationRecorder(self.model.layers, at=[mid_layer]) as rec:
                self.model.forward(input_ids)
                h = rec.activations[mid_layer][0].detach().float()

            # PCA to get PC1
            # (In production, cache the PC direction from a calibration set)
            pc1 = h.mean(dim=0)
            pc1 = pc1 / max(pc1.norm(), 1e-10)

            # Logit lens
            ll_logits = self.model.unembed(pc1.unsqueeze(0)).squeeze(0).float()
            ll_probs = torch.softmax(ll_logits, dim=-1)
            ll_topk = torch.topk(ll_probs, 10)
            dominant = [(self.model.tokenizer.decode([idx.item()]).strip(), prob.item())
                        for idx, prob in zip(ll_topk.indices, ll_topk.values)]

            # J-lens
            if mid_layer in self.lens.jacobians:
                transported = self.lens.transport(pc1.cpu().float().unsqueeze(0), mid_layer)
                jl_logits = self.model.unembed(transported.to(h.device)).squeeze(0).float()
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
                pc1_variance_pct=0.0,  # Would need full PCA to compute
                dominant_tokens=dominant[:5],
                secondary_tokens=secondary[:5],
                cosine_logit_jlens=cos,
            )
        except Exception:
            return None

    def _measure_loading(self, memory_id: str, content: str,
                         task: str, markers: list[str]) -> MemoryLoadingResult:
        """Check if memory markers reach workspace."""
        mem = MemoryProbe(
            memory_id=memory_id,
            content=content,
            marker_tokens=markers,
        )

        result = self.workspace_probe.probe(
            [mem], task, model_name=self.model_name)

        if result.memories:
            mr = result.memories[0]
            return MemoryLoadingResult(
                memory_id=memory_id,
                marker_tokens=markers,
                mean_workspace_rank=mr.mean_best_rank_ws if hasattr(mr, 'mean_best_rank_ws') else -1,
                baseline_rank=-1,  # Would need baseline run
                delta=0,
                loaded=mr.workspace_loaded if hasattr(mr, 'workspace_loaded') else False,
            )

        return MemoryLoadingResult(
            memory_id=memory_id,
            marker_tokens=markers,
            mean_workspace_rank=-1,
            baseline_rank=-1,
            delta=0,
            loaded=False,
        )

    def _find_onset(self, readings: list[JSpaceReading]) -> int:
        """Find the first workspace layer with content."""
        for r in sorted(readings, key=lambda x: x.layer):
            if r.in_workspace and r.top_tokens:
                return r.layer
        return -1

    def _dominant_tokens(self, readings: list[JSpaceReading]) -> list[str]:
        """Get the most common tokens across workspace readings."""
        from collections import Counter
        all_tokens = []
        for r in readings:
            for tok, _ in r.top_tokens[:5]:
                if tok.strip():
                    all_tokens.append(tok)
        return [tok for tok, _ in Counter(all_tokens).most_common(10)]
