"""Anchored Persistence — Geometry readings with semantic context.

Extends PersistentStore with three capabilities:
  1. Semantic anchoring: link geometry to the content that produced it
  2. Concept trajectories: track how geometry evolves for a concept over time
  3. MoE expert profiles: per-expert geometric signatures

The geometry becomes queryable by meaning, not just time.
"What does this model look like when discussing consciousness?"
instead of "What was the average rank over 24 hours?"

Backward compatible — existing record_geometry/get_trend still work.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import statistics
import time
from typing import Optional

try:
    from .persistence import PersistentStore
except ImportError:
    from persistence import PersistentStore

GEOMETRY_VOCAB = {
    "consciousness": "awareness experience qualia phenomenal subjective sentience",
    "deception": "lying dishonest misalignment confabulation hallucination untruthful",
    "emotion": "affect valence arousal sentiment feeling mood",
    "reasoning": "logic inference deduction chain-of-thought planning deliberation",
    "memory": "recall retrieval storage consolidation forgetting persistence",
    "creativity": "novel divergent generative imagination originality",
    "attention": "focus salience selection gating concentration",
    "uncertainty": "confidence calibration hedging doubt ambiguity",
    "self-reference": "introspection metacognition self-model identity self-awareness",
    "self-awareness": "introspection metacognition self-model identity self-reference",
    "refusal": "decline reject safety boundary guardrail alignment",
    "sycophancy": "agreement flattery compliance user-pleasing validation-seeking",
    "code": "programming implementation syntax debugging software engineering",
    "math": "arithmetic calculation proof theorem algebra geometry",
    "factual": "knowledge retrieval recall fact accuracy truth grounding",
    "roleplay": "persona character simulation acting performance",
    "ethics": "moral values principles right wrong consent harm",
    "kv cache": "key-value attention geometry eigenvalue spectral rank",
    "expert": "moe mixture routing specialization gating sparse",
}


def _enrich_prompt(text: str) -> str:
    """Generate SIRA-style search terms for a prompt."""
    text_lower = text.lower()
    terms = []
    for trigger, expansions in GEOMETRY_VOCAB.items():
        if trigger in text_lower:
            terms.extend(expansions.split())
    return " ".join(set(terms))


def _concept_hash(text: str, entities: list[str] = None) -> str:
    """Generate a stable hash for concept trajectory grouping.

    Normalizes by extracting key noun phrases and entities, sorting them,
    and hashing. Same concept with different phrasing should group together.
    """
    words = set(re.findall(r'\b[a-z]{3,}\b', text.lower()))
    stop = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all',
            'can', 'had', 'her', 'was', 'one', 'our', 'out', 'has',
            'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see',
            'way', 'who', 'did', 'get', 'let', 'say', 'she', 'too',
            'use', 'this', 'that', 'with', 'have', 'from', 'they',
            'been', 'said', 'each', 'which', 'their', 'will', 'what',
            'when', 'make', 'like', 'just', 'over', 'such', 'take',
            'than', 'them', 'very', 'some', 'could', 'into', 'about',
            'would', 'there', 'these', 'other', 'more'}
    content_words = sorted(words - stop)

    if entities:
        content_words.extend(sorted(e.lower() for e in entities))

    key = "|".join(content_words[:20])
    return hashlib.sha256(key.encode()).hexdigest()[:12]


def _extract_entities(text: str) -> list[str]:
    """Extract candidate entities from text (capitalized phrases, acronyms)."""
    entities = []
    for m in re.finditer(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', text):
        ent = m.group(1)
        if len(ent) > 2 and ent not in {'The', 'This', 'That', 'What', 'When',
                                          'Where', 'Which', 'Here', 'There'}:
            entities.append(ent)
    for m in re.finditer(r'\b([A-Z]{2,}(?:-[A-Z]+)*)\b', text):
        entities.append(m.group(1))
    return list(set(entities))


class AnchoredStore(PersistentStore):
    """Geometry persistence with semantic anchoring, trajectories, and MoE support."""

    def __init__(self, db_path: str = None):
        super().__init__(db_path or PersistentStore.__init__.__defaults__[0])
        self._init_anchored_schema()

    def _init_anchored_schema(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS geometry_anchors (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reading_id INTEGER NOT NULL,
                    prompt_text TEXT,
                    search_terms TEXT,
                    entities TEXT,
                    concept_hash TEXT,
                    session_id TEXT,
                    created_at REAL,
                    FOREIGN KEY (reading_id) REFERENCES geometry_readings(id)
                );

                CREATE TABLE IF NOT EXISTS expert_geometry (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reading_id INTEGER NOT NULL,
                    expert_id INTEGER NOT NULL,
                    layer INTEGER,
                    router_prob REAL,
                    effective_rank REAL,
                    spectral_entropy REAL,
                    norm_per_token REAL,
                    extra TEXT,
                    FOREIGN KEY (reading_id) REFERENCES geometry_readings(id)
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS anchor_fts USING fts5(
                    prompt_text, search_terms
                );

                CREATE INDEX IF NOT EXISTS idx_anchor_concept
                    ON geometry_anchors(concept_hash);
                CREATE INDEX IF NOT EXISTS idx_anchor_session
                    ON geometry_anchors(session_id);
                CREATE INDEX IF NOT EXISTS idx_expert_geo
                    ON expert_geometry(reading_id, expert_id);
            """)

    def record_anchored(self, snapshot_id: str, checkpoint: str,
                        geometry: dict, prompt_text: str,
                        entities: list[str] = None,
                        session_id: str = None,
                        expert_geometries: list[dict] = None) -> int:
        """Record geometry with semantic anchoring and optional per-expert data.

        Returns the reading_id for cross-referencing.
        """
        with self._conn() as conn:
            cur = conn.execute(
                """INSERT INTO geometry_readings
                   (snapshot_id, checkpoint, timestamp, effective_rank,
                    spectral_entropy, norm_per_token, top_sv_ratio,
                    key_norm, extra)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    snapshot_id, checkpoint, time.time(),
                    geometry.get("effective_rank"),
                    geometry.get("spectral_entropy"),
                    geometry.get("norm_per_token"),
                    geometry.get("top_sv_ratio"),
                    geometry.get("key_norm"),
                    json.dumps({
                        k: v for k, v in geometry.items()
                        if k not in ("effective_rank", "spectral_entropy",
                                     "norm_per_token", "top_sv_ratio", "key_norm")
                    }) if geometry else None,
                ),
            )
            reading_id = cur.lastrowid

            if entities is None:
                entities = _extract_entities(prompt_text)
            search_terms = _enrich_prompt(prompt_text)
            concept_h = _concept_hash(prompt_text, entities)

            conn.execute(
                """INSERT INTO geometry_anchors
                   (reading_id, prompt_text, search_terms, entities,
                    concept_hash, session_id, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (reading_id, prompt_text, search_terms,
                 json.dumps(entities), concept_h, session_id, time.time()),
            )

            anchor_id = conn.execute(
                "SELECT id FROM geometry_anchors WHERE reading_id = ?",
                (reading_id,)
            ).fetchone()["id"]

            conn.execute(
                "INSERT INTO anchor_fts(rowid, prompt_text, search_terms) VALUES (?, ?, ?)",
                (anchor_id, prompt_text, search_terms),
            )

            if expert_geometries:
                for eg in expert_geometries:
                    extra_fields = {k: v for k, v in eg.items()
                                    if k not in ("expert_id", "layer", "router_prob",
                                                 "effective_rank", "spectral_entropy",
                                                 "norm_per_token")}
                    conn.execute(
                        """INSERT INTO expert_geometry
                           (reading_id, expert_id, layer, router_prob,
                            effective_rank, spectral_entropy, norm_per_token, extra)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (reading_id, eg["expert_id"], eg.get("layer"),
                         eg.get("router_prob"), eg.get("effective_rank"),
                         eg.get("spectral_entropy"), eg.get("norm_per_token"),
                         json.dumps(extra_fields) if extra_fields else None),
                    )

        return reading_id

    def query_by_concept(self, concept_text: str, limit: int = 50) -> list[dict]:
        """Find geometry readings related to a concept via FTS5."""
        enriched = _enrich_prompt(concept_text)
        search_query = f"{concept_text} {enriched}".strip()

        fts_terms = " OR ".join(f'"{w}"' for w in search_query.split() if len(w) > 2)
        if not fts_terms:
            return []

        with self._conn() as conn:
            rows = conn.execute("""
                SELECT ga.id, ga.reading_id, ga.prompt_text, ga.concept_hash,
                       ga.session_id, ga.entities,
                       gr.effective_rank, gr.spectral_entropy,
                       gr.norm_per_token, gr.top_sv_ratio,
                       gr.key_norm, gr.timestamp, gr.snapshot_id,
                       anchor_fts.rank as relevance
                FROM anchor_fts
                JOIN geometry_anchors ga ON ga.id = anchor_fts.rowid
                JOIN geometry_readings gr ON gr.id = ga.reading_id
                WHERE anchor_fts MATCH ?
                ORDER BY anchor_fts.rank
                LIMIT ?
            """, (fts_terms, limit)).fetchall()

        return [
            {
                "reading_id": r["reading_id"],
                "prompt_text": r["prompt_text"],
                "concept_hash": r["concept_hash"],
                "session_id": r["session_id"],
                "entities": json.loads(r["entities"]) if r["entities"] else [],
                "geometry": {
                    "effective_rank": r["effective_rank"],
                    "spectral_entropy": r["spectral_entropy"],
                    "norm_per_token": r["norm_per_token"],
                    "top_sv_ratio": r["top_sv_ratio"],
                    "key_norm": r["key_norm"],
                },
                "timestamp": r["timestamp"],
                "snapshot_id": r["snapshot_id"],
                "relevance": r["relevance"],
            }
            for r in rows
        ]

    def get_trajectory(self, concept_text: str, days: int = 30) -> dict:
        """Track how geometry evolves for a concept over time.

        Groups readings by concept_hash, orders by time, computes
        drift metrics between readings of the same concept.
        """
        results = self.query_by_concept(concept_text, limit=500)
        if not results:
            return {
                "concept": concept_text, "readings": [],
                "drift": None, "total_readings": 0,
            }

        cutoff = time.time() - (days * 86400)
        results = [r for r in results if r["timestamp"] >= cutoff]
        if not results:
            return {
                "concept": concept_text, "readings": [],
                "drift": None, "total_readings": 0,
            }

        hash_groups = {}
        for r in results:
            h = r["concept_hash"]
            hash_groups.setdefault(h, []).append(r)

        largest_group = max(hash_groups.values(), key=len)
        largest_group.sort(key=lambda r: r["timestamp"])

        ranks = [r["geometry"]["effective_rank"] for r in largest_group
                 if r["geometry"]["effective_rank"] is not None]
        entropies = [r["geometry"]["spectral_entropy"] for r in largest_group
                     if r["geometry"]["spectral_entropy"] is not None]

        drift = {}
        if len(ranks) >= 2:
            drift["rank_trend"] = _linear_slope(ranks)
            drift["rank_std"] = statistics.stdev(ranks) if len(ranks) > 1 else 0
        if len(entropies) >= 2:
            drift["entropy_trend"] = _linear_slope(entropies)
            drift["entropy_std"] = statistics.stdev(entropies) if len(entropies) > 1 else 0

        if ranks and drift:
            mean_rank = statistics.mean(ranks)
            if mean_rank > 0:
                cv = (drift.get("rank_std", 0) / mean_rank)
                drift["stability"] = max(0, 1 - cv)
            else:
                drift["stability"] = 1.0

        return {
            "concept": concept_text,
            "readings": largest_group,
            "drift": drift if drift else None,
            "first_seen": largest_group[0]["timestamp"],
            "last_seen": largest_group[-1]["timestamp"],
            "total_readings": len(largest_group),
            "concept_hash": largest_group[0]["concept_hash"],
        }

    def get_expert_profile(self, expert_id: int, layer: int = None) -> dict:
        """Profile an expert's geometric behavior and associated content."""
        with self._conn() as conn:
            if layer is not None:
                rows = conn.execute("""
                    SELECT eg.*, ga.prompt_text, ga.entities, ga.concept_hash
                    FROM expert_geometry eg
                    JOIN geometry_anchors ga ON ga.reading_id = eg.reading_id
                    WHERE eg.expert_id = ? AND eg.layer = ?
                    ORDER BY eg.id DESC
                """, (expert_id, layer)).fetchall()
            else:
                rows = conn.execute("""
                    SELECT eg.*, ga.prompt_text, ga.entities, ga.concept_hash
                    FROM expert_geometry eg
                    JOIN geometry_anchors ga ON ga.reading_id = eg.reading_id
                    WHERE eg.expert_id = ?
                    ORDER BY eg.id DESC
                """, (expert_id,)).fetchall()

        if not rows:
            return {
                "expert_id": expert_id, "layer": layer,
                "activation_count": 0,
            }

        ranks = [r["effective_rank"] for r in rows if r["effective_rank"] is not None]
        entropies = [r["spectral_entropy"] for r in rows if r["spectral_entropy"] is not None]
        router_probs = [r["router_prob"] for r in rows if r["router_prob"] is not None]

        concept_counts = {}
        for r in rows:
            h = r["concept_hash"]
            if h:
                concept_counts[h] = concept_counts.get(h, 0) + 1

        top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        prompt_samples = []
        seen_hashes = set()
        for r in rows:
            h = r["concept_hash"]
            if h and h not in seen_hashes and r["prompt_text"]:
                prompt_samples.append(r["prompt_text"][:200])
                seen_hashes.add(h)
                if len(prompt_samples) >= 5:
                    break

        return {
            "expert_id": expert_id,
            "layer": layer,
            "activation_count": len(rows),
            "avg_geometry": {
                "effective_rank": statistics.mean(ranks) if ranks else None,
                "spectral_entropy": statistics.mean(entropies) if entropies else None,
            },
            "avg_router_prob": statistics.mean(router_probs) if router_probs else None,
            "concept_hashes": top_concepts,
            "sample_prompts": prompt_samples,
            "geometric_range": {
                "rank_min": min(ranks) if ranks else None,
                "rank_max": max(ranks) if ranks else None,
                "entropy_min": min(entropies) if entropies else None,
                "entropy_max": max(entropies) if entropies else None,
            },
        }

    def get_session_geometry(self, session_id: str) -> list[dict]:
        """Get all anchored readings for a session, ordered by time."""
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT ga.prompt_text, ga.entities, ga.concept_hash,
                       gr.effective_rank, gr.spectral_entropy,
                       gr.norm_per_token, gr.top_sv_ratio,
                       gr.key_norm, gr.timestamp, gr.snapshot_id, gr.checkpoint
                FROM geometry_anchors ga
                JOIN geometry_readings gr ON gr.id = ga.reading_id
                WHERE ga.session_id = ?
                ORDER BY gr.timestamp
            """, (session_id,)).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> dict:
        """Summary statistics for the anchored store."""
        with self._conn() as conn:
            total_readings = conn.execute(
                "SELECT COUNT(*) FROM geometry_readings"
            ).fetchone()[0]
            total_anchored = conn.execute(
                "SELECT COUNT(*) FROM geometry_anchors"
            ).fetchone()[0]
            total_experts = conn.execute(
                "SELECT COUNT(DISTINCT expert_id) FROM expert_geometry"
            ).fetchone()[0]
            unique_concepts = conn.execute(
                "SELECT COUNT(DISTINCT concept_hash) FROM geometry_anchors"
            ).fetchone()[0]
            sessions = conn.execute(
                "SELECT COUNT(DISTINCT session_id) FROM geometry_anchors WHERE session_id IS NOT NULL"
            ).fetchone()[0]

        return {
            "total_readings": total_readings,
            "anchored_readings": total_anchored,
            "unique_concepts": unique_concepts,
            "unique_experts": total_experts,
            "sessions": sessions,
        }


def _linear_slope(values: list[float]) -> float:
    """Compute slope of a linear fit over ordered values."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = statistics.mean(values)
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator
