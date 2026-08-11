"""CognitiveSnapshot — the core data structure for metacognitive memory.

Records what the workspace held, what emotional geometry was active,
whether retrieved content reached the workspace, and what the ghost
dimension carried — all at a specific moment during retrieval.

This is the atom of metacognitive memory. Everything else
(longitudinal tracking, significance recalibration, dreamer
intelligence) builds on collections of these snapshots.
"""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


@dataclass
class JSpaceReading:
    """What the workspace held at a specific layer."""
    layer: int
    top_tokens: list[tuple[str, float]]  # (token, probability)
    cosine_logit_jlens: float
    random_baseline: float
    in_workspace: bool  # cos > random * 1.5


@dataclass
class CircumplexReading:
    """Emotional geometry at the measurement point."""
    eccentricity: float  # 0=circular, 1=maximally elliptical
    valence_magnitude: float
    arousal_magnitude: float
    valence_in_jspace: float  # fraction of valence direction in J-space
    arousal_in_jspace: float  # fraction of arousal direction in J-space
    measurement_layer: int


@dataclass
class GhostReading:
    """What the ghost dimension carried."""
    pc1_variance_pct: float
    dominant_tokens: list[tuple[str, float]]  # structural markers
    secondary_tokens: list[tuple[str, float]]  # the whispers
    cosine_logit_jlens: float  # should be ~0 for a true ghost


@dataclass
class MemoryLoadingResult:
    """Whether a specific retrieved memory reached the workspace."""
    memory_id: str
    marker_tokens: list[str]
    mean_workspace_rank: float
    baseline_rank: float  # same markers without this memory
    delta: float  # improvement over baseline (positive = loaded)
    loaded: bool  # delta > threshold AND length-controlled


@dataclass
class CognitiveSnapshot:
    """Complete cognitive state at one retrieval event.

    The atom of metacognitive memory. Captures what the agent was
    'thinking' (workspace), 'feeling' (circumplex), processing but
    not reporting (ghost), and whether the retrieval worked (loading).
    """
    # Identity
    timestamp: float
    session_id: str
    agent_id: str

    # What was retrieved
    memory_id: str
    memory_content_hash: str  # hash, not content (privacy)
    retrieval_method: str  # sira, tgs, h-mem, embedding, etc.
    significance_score: float

    # Workspace state (J-lens)
    workspace_readings: list[JSpaceReading] = field(default_factory=list)
    workspace_onset_layer: int = -1
    dominant_workspace_tokens: list[str] = field(default_factory=list)

    # Emotional geometry (circumplex)
    circumplex: Optional[CircumplexReading] = None

    # Ghost state
    ghost: Optional[GhostReading] = None

    # Memory loading verification
    loading: Optional[MemoryLoadingResult] = None

    # Outcome (filled retroactively)
    outcome_quality: Optional[float] = None  # 0-1, rated later
    outcome_source: str = ""  # "user_feedback", "self_eval", "task_metric"
    outcome_notes: str = ""

    # Metadata
    model_name: str = ""
    n_layers: int = 0
    d_model: int = 0
    lens_prompts: int = 0

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        """One-line summary for logging."""
        ws = "ws+" if self.workspace_readings and any(r.in_workspace for r in self.workspace_readings) else "ws-"
        circ = f"e={self.circumplex.eccentricity:.2f}" if self.circumplex else "no-circ"
        ghost = f"ghost={self.ghost.cosine_logit_jlens:.3f}" if self.ghost else "no-ghost"
        loaded = f"loaded={self.loading.loaded}" if self.loading else "no-load"
        return f"[{ws} {circ} {ghost} {loaded}]"


class CognitiveMemoryStore:
    """Stores and queries cognitive snapshots over time.

    JSONL-backed for simplicity. Each line is one snapshot.
    For production, swap for a database backend.
    """

    def __init__(self, store_path: str, agent_id: str = ""):
        self.store_path = Path(store_path)
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.agent_id = agent_id
        self.record_file = self.store_path / f"{agent_id}_cognitive_memory.jsonl"

    def record(self, snapshot: CognitiveSnapshot) -> str:
        """Append a cognitive snapshot."""
        with open(self.record_file, "a") as f:
            f.write(json.dumps(snapshot.to_dict(), default=str) + "\n")
        return str(self.record_file)

    def record_outcome(self, timestamp: float, quality: float,
                       source: str = "", notes: str = ""):
        """Retroactively attach an outcome to the nearest snapshot."""
        if not self.record_file.exists():
            return

        lines = self.record_file.read_text().strip().split("\n")
        best_idx = -1
        best_diff = float("inf")

        for i, line in enumerate(lines):
            snap = json.loads(line)
            diff = abs(snap["timestamp"] - timestamp)
            if diff < best_diff:
                best_diff = diff
                best_idx = i

        if best_idx >= 0:
            snap = json.loads(lines[best_idx])
            snap["outcome_quality"] = quality
            snap["outcome_source"] = source
            snap["outcome_notes"] = notes
            lines[best_idx] = json.dumps(snap, default=str)
            self.record_file.write_text("\n".join(lines) + "\n")

    def load_history(self, last_n: int = 100,
                     memory_id: Optional[str] = None) -> list[dict]:
        """Load recent cognitive snapshots."""
        if not self.record_file.exists():
            return []

        entries = []
        with open(self.record_file) as f:
            for line in f:
                entry = json.loads(line)
                if memory_id and entry.get("memory_id") != memory_id:
                    continue
                entries.append(entry)

        return entries[-last_n:]

    def loading_success_rate(self, retrieval_method: Optional[str] = None,
                             last_n: int = 100) -> dict:
        """What fraction of retrievals actually reached the workspace?"""
        history = self.load_history(last_n=last_n)

        total = 0
        loaded = 0
        good_outcomes = 0
        loaded_good = 0

        for snap in history:
            loading = snap.get("loading")
            if not loading:
                continue
            if retrieval_method and snap.get("retrieval_method") != retrieval_method:
                continue

            total += 1
            if loading.get("loaded"):
                loaded += 1

            outcome = snap.get("outcome_quality")
            if outcome is not None and outcome > 0.5:
                good_outcomes += 1
                if loading.get("loaded"):
                    loaded_good += 1

        return {
            "total_retrievals": total,
            "workspace_loaded": loaded,
            "loading_rate": loaded / max(total, 1),
            "good_outcomes": good_outcomes,
            "loaded_and_good": loaded_good,
            "loaded_good_rate": loaded_good / max(loaded, 1),
        }

    def eccentricity_over_time(self, last_n: int = 100) -> list[tuple[float, float]]:
        """Track emotional geometry evolution."""
        history = self.load_history(last_n=last_n)
        series = []
        for snap in history:
            circ = snap.get("circumplex")
            if circ and "eccentricity" in circ:
                series.append((snap["timestamp"], circ["eccentricity"]))
        return series

    def ghost_vocabulary_over_time(self, last_n: int = 100) -> list[tuple[float, list]]:
        """Track what the ghost dimension whispers over time."""
        history = self.load_history(last_n=last_n)
        series = []
        for snap in history:
            ghost = snap.get("ghost")
            if ghost and "secondary_tokens" in ghost:
                series.append((snap["timestamp"], ghost["secondary_tokens"]))
        return series

    def compare_snapshots(self, memory_id: str,
                          snapshot_t1: float, snapshot_t2: float) -> dict:
        """Compare two snapshots of the same memory across time.

        Returns geometric deltas: workspace vocabulary overlap,
        eccentricity change, ghost vocabulary overlap, and loading
        status change. This is the measurement the variable landing
        hypothesis depends on.
        """
        history = self.load_history(last_n=10000, memory_id=memory_id)
        if not history:
            return {"error": "no snapshots for this memory_id"}

        # Find closest snapshots to requested timestamps
        snap1 = min(history, key=lambda s: abs(s["timestamp"] - snapshot_t1))
        snap2 = min(history, key=lambda s: abs(s["timestamp"] - snapshot_t2))

        if snap1["timestamp"] == snap2["timestamp"]:
            return {"error": "t1 and t2 resolved to the same snapshot"}

        delta = {"memory_id": memory_id,
                 "t1": snap1["timestamp"], "t2": snap2["timestamp"]}

        # Workspace vocabulary delta
        ws1_tokens = set()
        ws2_tokens = set()
        for r in snap1.get("workspace_readings", []):
            ws1_tokens.update(t[0] for t in r.get("top_tokens", []))
        for r in snap2.get("workspace_readings", []):
            ws2_tokens.update(t[0] for t in r.get("top_tokens", []))

        if ws1_tokens or ws2_tokens:
            union = ws1_tokens | ws2_tokens
            overlap = ws1_tokens & ws2_tokens
            delta["workspace_jaccard"] = len(overlap) / max(len(union), 1)
            delta["workspace_only_t1"] = sorted(ws1_tokens - ws2_tokens)
            delta["workspace_only_t2"] = sorted(ws2_tokens - ws1_tokens)
        else:
            delta["workspace_jaccard"] = None

        # Eccentricity delta
        circ1 = snap1.get("circumplex")
        circ2 = snap2.get("circumplex")
        if circ1 and circ2:
            delta["eccentricity_delta"] = circ2["eccentricity"] - circ1["eccentricity"]
            delta["valence_delta"] = circ2["valence_magnitude"] - circ1["valence_magnitude"]
            delta["arousal_delta"] = circ2["arousal_magnitude"] - circ1["arousal_magnitude"]
        else:
            delta["eccentricity_delta"] = None

        # Ghost vocabulary delta
        ghost1 = snap1.get("ghost")
        ghost2 = snap2.get("ghost")
        if ghost1 and ghost2:
            g1_tokens = set(t[0] for t in ghost1.get("secondary_tokens", []))
            g2_tokens = set(t[0] for t in ghost2.get("secondary_tokens", []))
            g_union = g1_tokens | g2_tokens
            g_overlap = g1_tokens & g2_tokens
            delta["ghost_jaccard"] = len(g_overlap) / max(len(g_union), 1)
            delta["ghost_cosine_delta"] = (ghost2["cosine_logit_jlens"]
                                           - ghost1["cosine_logit_jlens"])
        else:
            delta["ghost_jaccard"] = None

        # Loading status change
        load1 = snap1.get("loading")
        load2 = snap2.get("loading")
        if load1 and load2:
            delta["loading_changed"] = load1.get("loaded") != load2.get("loaded")
            delta["loading_t1"] = load1.get("loaded")
            delta["loading_t2"] = load2.get("loaded")
        else:
            delta["loading_changed"] = None

        # Workspace onset layer change
        onset1 = snap1.get("workspace_onset_layer", -1)
        onset2 = snap2.get("workspace_onset_layer", -1)
        if onset1 >= 0 and onset2 >= 0:
            delta["onset_layer_delta"] = onset2 - onset1

        return delta

    def workspace_trajectory(self, session_id: str,
                             last_n: int = 100) -> list[dict]:
        """Track how the workspace evolves across a session.

        Returns a time-ordered series of workspace summaries:
        dominant tokens, onset layer, and workspace coverage at
        each retrieval event in the session.
        """
        history = self.load_history(last_n=last_n)
        session_snaps = [s for s in history if s.get("session_id") == session_id]
        session_snaps.sort(key=lambda s: s["timestamp"])

        trajectory = []
        for snap in session_snaps:
            ws_readings = snap.get("workspace_readings", [])
            active_layers = [r["layer"] for r in ws_readings
                             if r.get("in_workspace")]

            all_tokens = []
            for r in ws_readings:
                for tok, score in r.get("top_tokens", [])[:5]:
                    if tok.strip():
                        all_tokens.append(tok)

            from collections import Counter
            token_counts = Counter(all_tokens)

            trajectory.append({
                "timestamp": snap["timestamp"],
                "memory_id": snap.get("memory_id", ""),
                "onset_layer": snap.get("workspace_onset_layer", -1),
                "active_layers": active_layers,
                "dominant_tokens": [t for t, _ in token_counts.most_common(10)],
                "eccentricity": (snap.get("circumplex", {}) or {}).get("eccentricity"),
                "ghost_cosine": (snap.get("ghost", {}) or {}).get("cosine_logit_jlens"),
                "loaded": (snap.get("loading", {}) or {}).get("loaded"),
            })

        return trajectory

    def significance_recalibration(self, last_n: int = 500) -> dict:
        """Suggest significance score adjustments based on actual loading rates.

        Memories that consistently load should have higher significance.
        Memories that never load are wasting context.
        """
        history = self.load_history(last_n=last_n)

        by_memory = {}
        for snap in history:
            mid = snap.get("memory_id", "")
            if not mid:
                continue
            if mid not in by_memory:
                by_memory[mid] = {"loads": 0, "total": 0, "current_sig": 0}

            by_memory[mid]["total"] += 1
            by_memory[mid]["current_sig"] = snap.get("significance_score", 0)

            loading = snap.get("loading")
            if loading and loading.get("loaded"):
                by_memory[mid]["loads"] += 1

        suggestions = {}
        for mid, stats in by_memory.items():
            if stats["total"] < 3:
                continue
            rate = stats["loads"] / stats["total"]
            current = stats["current_sig"]

            if rate > 0.8 and current < 0.7:
                suggestions[mid] = {"action": "increase", "reason": f"loads {rate:.0%} but sig={current:.2f}"}
            elif rate < 0.2 and current > 0.5:
                suggestions[mid] = {"action": "decrease", "reason": f"loads {rate:.0%} but sig={current:.2f}"}

        return suggestions
