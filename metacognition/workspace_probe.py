"""Mnemosyne J-lens workspace probe — measures whether retrieved memories reach J-space.

Uses pinned_token_ids for exact ranks at every (position, layer) cell.

v3: Fixed tokenizer artifact (single-char subtokens), added baseline control,
    uses only whole-word tokens as markers.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import jlens
from jlens.hf import HFLensModel
from jlens.vis import compute_slice, SliceData


@dataclass
class MemoryProbe:
    """A single memory item to probe for workspace presence."""
    memory_id: str
    content: str
    marker_tokens: list[str]
    significance: float = 0.5
    source: str = ""


@dataclass
class ConceptLoadingResult:
    """Full rank profile of a concept across layers."""
    token_str: str
    token_id: int
    is_single_token: bool           # True = whole word is one token (reliable)
    vocab_size: int
    rank_by_layer: dict             # layer_num -> mean rank across task positions
    best_rank: int = -1
    best_layer: int = -1
    best_position: int = -1
    mean_rank_sensory: float = -1
    mean_rank_workspace: float = -1
    rank_improvement: float = 0.0
    workspace_loaded: bool = False
    sensory_only: bool = False


@dataclass
class MemoryLoadingResult:
    """Full J-space loading result for one memory."""
    memory_id: str
    workspace_loaded: bool
    n_concepts_ws_loaded: int
    n_concepts_total: int
    n_reliable_concepts: int        # single-token markers only
    workspace_rate: float
    concept_results: list[ConceptLoadingResult] = field(default_factory=list)


@dataclass
class ProbeResult:
    """Complete result of a workspace probe measurement."""
    timestamp: float
    model_name: str
    prompt_tokens: int
    task_positions: list[int]
    n_layers: int
    layers_measured: list[int]
    vocab_size: int
    workspace_onset_layer: int
    rank_threshold: int
    memories: list[MemoryLoadingResult] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def summary(self) -> str:
        ws = [m for m in self.memories if m.workspace_loaded]
        return (
            f"{len(ws)}/{len(self.memories)} workspace-loaded | "
            f"threshold=top-{self.rank_threshold}/{self.vocab_size}"
        )

    def to_dict(self) -> dict:
        return asdict(self)


class WorkspaceProbe:
    """Measures whether retrieved memories reach the global workspace via J-lens."""

    def __init__(self, model: HFLensModel, lens: jlens.JacobianLens,
                 task_window: int = 8,
                 workspace_onset_frac: float = 0.46,
                 rank_threshold: int = 500):
        self.model = model
        self.lens = lens
        self.tokenizer = model.tokenizer
        self.task_window = task_window
        self.workspace_onset = int(model.n_layers * workspace_onset_frac)
        self.rank_threshold = rank_threshold

    def _tokenize_marker(self, marker: str) -> tuple[int, bool]:
        """Tokenize a marker and return (token_id, is_single_token).

        Only single-token markers are reliable. Multi-token markers
        decompose into subwords (often single characters) that produce
        artifacts — e.g. "sourdough" → ["s","ourd","ough"] and "s"
        appears at rank 0 in early layers because it's a common character.
        """
        ids = self.tokenizer.encode(marker, add_special_tokens=False)
        if len(ids) == 1:
            return ids[0], True
        # For multi-token markers, find the longest subtoken (most distinctive)
        decoded = [(tid, self.tokenizer.decode([tid])) for tid in ids]
        best = max(decoded, key=lambda x: len(x[1].strip()))
        # Skip if the best subtoken is a single character
        if len(best[1].strip()) <= 1:
            return -1, False
        return best[0], False

    def _collect_pins(self, memories: list[MemoryProbe]) -> tuple[set[int], dict]:
        """Gather pin IDs and marker-to-token mapping."""
        pin_ids = set()
        marker_map = {}  # (memory_id, marker_str) -> (token_id, is_single)
        for mem in memories:
            for marker in mem.marker_tokens:
                tid, is_single = self._tokenize_marker(marker)
                if tid >= 0:
                    pin_ids.add(tid)
                    marker_map[(mem.memory_id, marker)] = (tid, is_single)
        return pin_ids, marker_map

    def probe(self, memories: list[MemoryProbe], task_prompt: str,
              context_template: str = "Context:\n{context}\n\nQuestion: {task}\nAnswer:",
              model_name: str = "",
              extra_pin_ids: set[int] | None = None) -> ProbeResult:
        """Run a full workspace probe with exact rank tracking."""
        if memories:
            context = "\n".join(f"- {m.content}" for m in memories)
            full_prompt = context_template.format(context=context, task=task_prompt)
        else:
            full_prompt = f"Question: {task_prompt}\nAnswer:"

        all_pins, marker_map = self._collect_pins(memories)
        if extra_pin_ids:
            all_pins |= extra_pin_ids

        slice_data = compute_slice(
            self.model, self.lens, full_prompt,
            top_n=10,
            pinned_token_ids=all_pins,
            max_seq_len=512,
        )

        n_positions = slice_data.seq_len
        task_start = max(0, n_positions - self.task_window)
        task_positions = list(range(task_start, n_positions))

        tracked_ids = slice_data.tracked_token_ids
        id_to_idx = {tid: i for i, tid in enumerate(tracked_ids)}

        vocab_size = slice_data.vocab_size or self.tokenizer.vocab_size

        memory_results = []
        for mem in memories:
            concept_results = []
            for marker in mem.marker_tokens:
                key = (mem.memory_id, marker)
                if key not in marker_map:
                    continue
                tid, is_single = marker_map[key]
                tracked_idx = id_to_idx.get(tid)

                rank_by_layer = {}
                best_rank = vocab_size
                best_layer = -1
                best_pos = -1

                if tracked_idx is not None:
                    for layer_idx, layer_num in enumerate(slice_data.layers):
                        pos_ranks = []
                        for pos in task_positions:
                            if pos >= n_positions:
                                continue
                            r = int(slice_data.rank_tensor[pos, layer_idx, tracked_idx])
                            if r >= 0:
                                pos_ranks.append(r)
                                if r < best_rank:
                                    best_rank = r
                                    best_layer = layer_num
                                    best_pos = pos + slice_data.ctx_offset
                        if pos_ranks:
                            rank_by_layer[layer_num] = float(np.mean(pos_ranks))

                sensory_ranks = [v for k, v in rank_by_layer.items()
                                 if k < self.workspace_onset]
                ws_ranks = [v for k, v in rank_by_layer.items()
                            if k >= self.workspace_onset]

                mean_sensory = float(np.mean(sensory_ranks)) if sensory_ranks else -1
                mean_ws = float(np.mean(ws_ranks)) if ws_ranks else -1
                rank_imp = (mean_sensory - mean_ws) if (mean_sensory > 0 and mean_ws > 0) else 0

                ws_loaded = mean_ws > 0 and mean_ws < self.rank_threshold

                concept_results.append(ConceptLoadingResult(
                    token_str=marker,
                    token_id=tid,
                    is_single_token=is_single,
                    vocab_size=vocab_size,
                    rank_by_layer=rank_by_layer,
                    best_rank=best_rank if best_rank < vocab_size else -1,
                    best_layer=best_layer,
                    best_position=best_pos,
                    mean_rank_sensory=mean_sensory,
                    mean_rank_workspace=mean_ws,
                    rank_improvement=rank_imp,
                    workspace_loaded=ws_loaded,
                    sensory_only=(mean_sensory > 0 and mean_sensory < self.rank_threshold
                                  and not ws_loaded),
                ))

            reliable = [c for c in concept_results if c.is_single_token]
            n_ws = sum(1 for c in reliable if c.workspace_loaded)

            memory_results.append(MemoryLoadingResult(
                memory_id=mem.memory_id,
                workspace_loaded=n_ws > 0,
                n_concepts_ws_loaded=n_ws,
                n_concepts_total=len(concept_results),
                n_reliable_concepts=len(reliable),
                workspace_rate=n_ws / len(reliable) if reliable else 0,
                concept_results=concept_results,
            ))

        input_ids = self.model.encode(full_prompt, max_length=512)
        return ProbeResult(
            timestamp=time.time(),
            model_name=model_name,
            prompt_tokens=input_ids.shape[1],
            task_positions=[p + slice_data.ctx_offset for p in task_positions],
            n_layers=self.model.n_layers,
            layers_measured=slice_data.layers,
            vocab_size=vocab_size,
            workspace_onset_layer=self.workspace_onset,
            rank_threshold=self.rank_threshold,
            memories=memory_results,
            metadata={"task_prompt": task_prompt, "n_memories": len(memories)},
        )

    def probe_with_baseline(self, memories: list[MemoryProbe], task_prompt: str,
                            model_name: str = "") -> dict:
        """Run retrieval probe AND baseline (same markers, no context).

        The baseline answers: would these concepts appear at these layers
        just from the question, without the memory being injected?
        """
        _, marker_map = self._collect_pins(memories)
        all_tids = {tid for tid, _ in marker_map.values() if tid >= 0}

        with_ctx = self.probe(memories, task_prompt, model_name=model_name)

        # Baseline: same question, no memories, but SAME marker tokens pinned
        baseline_prompt = f"Question: {task_prompt}\nAnswer:"
        slice_data = compute_slice(
            self.model, self.lens, baseline_prompt,
            top_n=10, pinned_token_ids=all_tids, max_seq_len=512,
        )

        n_positions = slice_data.seq_len
        task_start = max(0, n_positions - self.task_window)
        task_positions = list(range(task_start, n_positions))
        tracked_ids = slice_data.tracked_token_ids
        id_to_idx = {tid: i for i, tid in enumerate(tracked_ids)}
        vocab_size = slice_data.vocab_size or self.tokenizer.vocab_size

        baseline_ranks = {}
        for (mem_id, marker), (tid, is_single) in marker_map.items():
            tracked_idx = id_to_idx.get(tid)
            if tracked_idx is None:
                continue
            rank_by_layer = {}
            for layer_idx, layer_num in enumerate(slice_data.layers):
                pos_ranks = []
                for pos in task_positions:
                    if pos >= n_positions:
                        continue
                    r = int(slice_data.rank_tensor[pos, layer_idx, tracked_idx])
                    if r >= 0:
                        pos_ranks.append(r)
                if pos_ranks:
                    rank_by_layer[layer_num] = float(np.mean(pos_ranks))
            baseline_ranks[(mem_id, marker)] = {
                "token_id": tid,
                "is_single_token": is_single,
                "rank_by_layer": rank_by_layer,
            }

        # Compute deltas: rank_with_memory - rank_without_memory
        # Negative delta = memory made concept rank BETTER (lower rank)
        deltas = {}
        for key, bl in baseline_ranks.items():
            mem_id, marker = key
            with_mem = None
            for mem_r in with_ctx.memories:
                if mem_r.memory_id == mem_id:
                    for c in mem_r.concept_results:
                        if c.token_str == marker:
                            with_mem = c
                            break
            if with_mem is None:
                continue

            layer_deltas = {}
            for layer_num, bl_rank in bl["rank_by_layer"].items():
                with_rank = with_mem.rank_by_layer.get(str(layer_num) if isinstance(
                    list(with_mem.rank_by_layer.keys())[0], str) else layer_num, None) if with_mem.rank_by_layer else None
                # Handle both string and int keys
                wr = with_mem.rank_by_layer.get(layer_num)
                if wr is None:
                    wr = with_mem.rank_by_layer.get(str(layer_num))
                if wr is not None:
                    layer_deltas[layer_num] = bl_rank - wr  # positive = memory helped

            ws_deltas = [v for k, v in layer_deltas.items() if k >= self.workspace_onset]
            sen_deltas = [v for k, v in layer_deltas.items() if k < self.workspace_onset]

            deltas[f"{mem_id}/{marker}"] = {
                "is_single_token": bl["is_single_token"],
                "mean_ws_improvement": float(np.mean(ws_deltas)) if ws_deltas else 0,
                "mean_sensory_improvement": float(np.mean(sen_deltas)) if sen_deltas else 0,
                "layer_deltas": layer_deltas,
            }

        return {
            "with_retrieval": with_ctx,
            "baseline_ranks": {f"{k[0]}/{k[1]}": v for k, v in baseline_ranks.items()},
            "retrieval_vs_baseline": deltas,
        }


class LongitudinalRecorder:
    """Records probe results over time for longitudinal study."""

    def __init__(self, store_path: str, agent_id: str = ""):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.agent_id = agent_id

    def record(self, result: ProbeResult, session_id: str = "",
               retrieval_method: str = "sira") -> str:
        entry = {
            "agent_id": self.agent_id,
            "session_id": session_id,
            "retrieval_method": retrieval_method,
            **result.to_dict(),
        }
        record_file = self.store_path / "probe_log.jsonl"
        with open(record_file, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
        return str(record_file)

    def load_history(self, memory_id: Optional[str] = None,
                     last_n: int = 100) -> list[dict]:
        record_file = self.store_path / "probe_log.jsonl"
        if not record_file.exists():
            return []
        entries = []
        with open(record_file) as f:
            for line in f:
                entry = json.loads(line)
                if memory_id:
                    mem_results = [m for m in entry.get("memories", [])
                                  if m["memory_id"] == memory_id]
                    if not mem_results:
                        continue
                entries.append(entry)
        return entries[-last_n:]

    def loading_rate_over_time(self, memory_id: str) -> list[tuple[float, float]]:
        history = self.load_history(memory_id=memory_id)
        series = []
        for entry in history:
            for mem in entry.get("memories", []):
                if mem["memory_id"] == memory_id:
                    series.append((entry["timestamp"], mem["workspace_rate"]))
        return series
