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

try:
    from .cognitive_snapshot import (
        CognitiveSnapshot, JSpaceReading, GhostReading,
        MemoryLoadingResult, CognitiveMemoryStore,
    )
    from .workspace_probe import WorkspaceProbe, MemoryProbe
    from .circumplex_probe import CircumplexProbe
    from .ghost_probe_class import GhostProbe
except ImportError:
    from cognitive_snapshot import (
        CognitiveSnapshot, JSpaceReading, GhostReading,
        MemoryLoadingResult, CognitiveMemoryStore,
    )
    from workspace_probe import WorkspaceProbe, MemoryProbe
    from circumplex_probe import CircumplexProbe
    from ghost_probe_class import GhostProbe


class MetacognitiveObserver:
    """Observes and records the cognitive state during retrieval events.

    Attach to a Mnemosyne instance to enable metacognitive memory.
    Does not modify retrieval behavior — purely observational.
    """

    #: Fallback random-direction cosine baseline (pre-calibrated for d~5120).
    #: Used only when the live matched-norm baseline cannot be computed.
    DEFAULT_RANDOM_BASELINE = 0.1

    def __init__(self, model: HFLensModel, lens: jlens.JacobianLens,
                 store_path: str, agent_id: str,
                 workspace_layers: Optional[list[int]] = None,
                 circumplex_layer: Optional[int] = None):
        self.model = model
        self.lens = lens
        self.agent_id = agent_id

        self.workspace_probe = WorkspaceProbe(model, lens)
        self.circumplex_probe = CircumplexProbe(model, lens)
        self.ghost_probe = GhostProbe(model, lens, probe_layers=[model.n_layers // 2])
        self.store = CognitiveMemoryStore(store_path, agent_id)

        self.workspace_layers = workspace_layers or [35, 39, 43, 45, 47]
        self.circumplex_layer = circumplex_layer or 45
        self.model_name = ""

    def observe_retrieval(self, memory_id: str, memory_content: str,
                          task_prompt: str, retrieval_method: str = "sira",
                          significance: float = 0.5,
                          session_id: str = "",
                          marker_tokens: Optional[list[str]] = None,
                          prior_context: str = "") -> CognitiveSnapshot:
        """Record a full cognitive snapshot at a retrieval event.

        Call this after SIRA (or any retriever) returns a memory,
        before the memory is used in generation.

        prior_context: accumulated conversational context to prepend.
        Used by variable landing experiment to test whether retrieval
        geometry changes when the system carries experiential history.
        """
        timestamp = time.time()

        # 1. Workspace readings at key layers
        ws_readings = self._measure_workspace(memory_content, task_prompt,
                                              prior_context=prior_context)

        # Build conversation text for live probes
        conversation_text = ""
        if prior_context:
            conversation_text = f"{prior_context}\n{memory_content}\n{task_prompt}"
        else:
            conversation_text = f"{memory_content}\n{task_prompt}"

        # 2. Circumplex at the ignition layer (live if calibrated)
        circ = self._measure_circumplex_live(conversation_text)

        # 3. Ghost state (live if calibrated)
        ghost = self._measure_ghost(conversation_text)

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

    def _measure_workspace(self, context: str, task: str,
                            prior_context: str = "") -> list[JSpaceReading]:
        """J-lens readings at workspace layers."""
        from jlens.vis import compute_slice

        if prior_context:
            prompt = f"Conversation history:\n{prior_context}\n\nContext:\n- {context}\n\nQuestion: {task}\nAnswer:"
        else:
            prompt = f"Context:\n- {context}\n\nQuestion: {task}\nAnswer:"
        slice_data = compute_slice(
            self.model, self.lens, prompt,
            top_n=10, max_seq_len=512,
        )

        # Real logit-lens vs J-lens agreement per workspace layer, plus a
        # matched-norm random baseline. Requires one extra forward pass
        # (SliceData does not retain activations).
        measured = [l for l in slice_data.layers if l in self.workspace_layers]
        cosines = self._compute_workspace_cosines(prompt, measured)

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

            # If the cosine could not be measured for this layer, fall back
            # to cos=0.0 against the pre-calibrated baseline, which yields
            # in_workspace=False — never a fabricated positive.
            cos, rand = cosines.get(
                layer_num, (0.0, self.DEFAULT_RANDOM_BASELINE))

            readings.append(JSpaceReading(
                layer=layer_num,
                top_tokens=tokens,
                cosine_logit_jlens=cos,
                random_baseline=rand,
                in_workspace=cos > rand * 1.5,
            ))

        return readings

    def _compute_workspace_cosines(
            self, prompt: str, layers: list[int],
            max_seq_len: int = 512) -> dict[int, tuple[float, float]]:
        """Measure logit-lens vs J-lens agreement at the last position.

        For each layer:
        1. Capture the residual h via ActivationRecorder.
        2. Transport through the J-lens: J_l @ h.
        3. cosine_logit_jlens = cosine(softmax(unembed(h)),
                                       softmax(unembed(J_l @ h)))
        4. random_baseline = cosine(softmax(unembed(h)),
                                    softmax(unembed(r))) where r is a
           random direction scaled to ||h||.

        Returns {layer: (cosine_logit_jlens, random_baseline)}. Layers the
        lens has no Jacobian for (e.g. the final layer, where J = I) and
        layers that fail to measure are omitted; the caller treats missing
        layers as not-in-workspace.
        """
        from jlens.hooks import ActivationRecorder

        fitted = [l for l in layers if l in self.lens.jacobians]
        result: dict[int, tuple[float, float]] = {}
        if not fitted:
            return result

        try:
            input_ids = self.model.encode(prompt, max_length=max_seq_len)
            with torch.no_grad():
                with ActivationRecorder(self.model.layers, at=fitted) as rec:
                    self.model.forward(input_ids)
                    activations = {
                        l: rec.activations[l].detach() for l in fitted}

                for layer in fitted:
                    h = activations[layer][0, -1, :].float()  # [d_model]

                    transported = self.lens.transport(h, layer)
                    ll_logits = self.model.unembed(h).float()
                    jl_logits = self.model.unembed(transported).float()

                    p_ll = torch.softmax(ll_logits, dim=-1)
                    p_jl = torch.softmax(jl_logits, dim=-1)
                    cos = float(torch.nn.functional.cosine_similarity(
                        p_ll, p_jl, dim=-1))

                    # Chance-level agreement: a random direction of
                    # matched norm through the same readout.
                    r = torch.randn_like(h)
                    r = r * (h.norm() / (r.norm() + 1e-8))
                    p_rand = torch.softmax(
                        self.model.unembed(r).float(), dim=-1)
                    rand = float(torch.nn.functional.cosine_similarity(
                        p_ll, p_rand, dim=-1))

                    result[layer] = (cos, rand)
        except Exception:
            # Measurement failure must not break retrieval; unmeasured
            # layers simply read as not-in-workspace.
            pass

        return result


    def calibrate_probes(self, prompts: Optional[list[str]] = None):
        """Calibrate ghost and circumplex probes. Call once per session."""
        self.ghost_probe.calibrate(prompts)
        self.circumplex_probe.calibrate(self.circumplex_layer)

    def calibrate_ghost_probe(self, prompts: Optional[list[str]] = None):
        """Calibrate the ghost probe with diverse prompts. Call once per session."""
        self.ghost_probe.calibrate(prompts)

    def _measure_ghost(self, conversation_text: str = "") -> Optional[GhostReading]:
        """Ghost dimension state at mid-network.

        If conversation_text is provided and probes are calibrated, measures
        LIVE ghost state from the conversation. Otherwise falls back to
        cached calibration measurement.
        """
        if not self.ghost_probe.is_calibrated:
            self.ghost_probe.calibrate()
        if conversation_text:
            return self.ghost_probe.measure_live(conversation_text)
        return self.ghost_probe.measure()

    def _measure_circumplex_live(self, conversation_text: str = ""):
        """Circumplex reading from live conversation text."""
        if conversation_text and getattr(self.circumplex_probe, '_calibrated', False):
            return self.circumplex_probe.measure_live(conversation_text)
        return self._measure_circumplex_static()

    def _measure_circumplex_static(self):
        """Original circumplex measurement (benchmark mode)."""
        try:
            result = self.circumplex_probe.measure_at_layer(self.circumplex_layer)
            return self.circumplex_probe.to_snapshot_reading(result)
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
