"""Tests for Anchored Persistence — semantic geometry, trajectories, MoE profiles."""

import json
import os
import sys
import tempfile
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from anchored import (
    AnchoredStore, _enrich_prompt, _concept_hash,
    _extract_entities, _linear_slope,
)
from persistence import PersistentStore


SAMPLE_GEOMETRY = {
    "effective_rank": 60.8,
    "spectral_entropy": 20.2,
    "norm_per_token": 3.5,
    "top_sv_ratio": 0.15,
    "key_norm": 42.0,
}


class TestEnrichPrompt:
    def test_consciousness_vocab(self):
        terms = _enrich_prompt("discussing consciousness and self-awareness")
        assert "qualia" in terms
        assert "phenomenal" in terms
        assert "introspection" in terms

    def test_no_match(self):
        terms = _enrich_prompt("the weather is nice today")
        assert terms == ""

    def test_multiple_domains(self):
        terms = _enrich_prompt("deception detection in emotional reasoning")
        assert "lying" in terms or "dishonest" in terms
        assert "affect" in terms or "valence" in terms
        assert "logic" in terms or "inference" in terms


class TestConceptHash:
    def test_deterministic(self):
        h1 = _concept_hash("consciousness and qualia")
        h2 = _concept_hash("consciousness and qualia")
        assert h1 == h2

    def test_different_concepts(self):
        h1 = _concept_hash("consciousness and qualia")
        h2 = _concept_hash("sewing machine USB protocol")
        assert h1 != h2

    def test_entities_included(self):
        h1 = _concept_hash("the model shows deception")
        h2 = _concept_hash("the model shows deception", ["Oracle", "Lyra"])
        assert h1 != h2

    def test_word_order_invariant(self):
        h1 = _concept_hash("consciousness qualia experience")
        h2 = _concept_hash("experience qualia consciousness")
        assert h1 == h2


class TestExtractEntities:
    def test_capitalized_phrases(self):
        entities = _extract_entities("Thomas Edrington built Oracle")
        assert "Thomas Edrington" in entities
        assert "Oracle" in entities

    def test_acronyms(self):
        entities = _extract_entities("The SIRA enrichment improved TGS-RAG retrieval")
        assert "SIRA" in entities

    def test_filters_common_words(self):
        entities = _extract_entities("The quick brown fox")
        assert "The" not in entities


class TestLinearSlope:
    def test_increasing(self):
        assert _linear_slope([1, 2, 3, 4, 5]) > 0

    def test_decreasing(self):
        assert _linear_slope([5, 4, 3, 2, 1]) < 0

    def test_flat(self):
        assert _linear_slope([3, 3, 3, 3]) == 0

    def test_single_value(self):
        assert _linear_slope([5]) == 0

    def test_empty(self):
        assert _linear_slope([]) == 0


class TestAnchoredStore:
    def _make_store(self, tmp):
        return AnchoredStore(os.path.join(tmp, "test.db"))

    def test_backward_compat(self):
        """Existing record_geometry still works."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_geometry("snap1", "turn_1", SAMPLE_GEOMETRY)
            history = store.get_geometry_history(10)
            assert len(history) == 1
            assert history[0]["effective_rank"] == 60.8

    def test_record_anchored(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            rid = store.record_anchored(
                "snap1", "turn_1", SAMPLE_GEOMETRY,
                "discussing consciousness and self-awareness",
                session_id="session_001",
            )
            assert rid > 0

            s = store.stats()
            assert s["total_readings"] == 1
            assert s["anchored_readings"] == 1

    def test_query_by_concept(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_anchored(
                "snap1", "turn_1", SAMPLE_GEOMETRY,
                "consciousness and qualia research",
            )
            store.record_anchored(
                "snap2", "turn_2", SAMPLE_GEOMETRY,
                "sewing machine USB protocol debugging",
            )

            results = store.query_by_concept("consciousness")
            assert len(results) >= 1
            assert "consciousness" in results[0]["prompt_text"].lower()

    def test_query_via_enriched_vocab(self):
        """SIRA vocabulary makes geometry findable via synonyms."""
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_anchored(
                "snap1", "turn_1", SAMPLE_GEOMETRY,
                "discussing consciousness in the model",
            )

            results = store.query_by_concept("qualia")
            assert len(results) >= 1

    def test_trajectory(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            for i in range(5):
                geo = dict(SAMPLE_GEOMETRY)
                geo["effective_rank"] = 50 + i * 5
                store.record_anchored(
                    f"snap_{i}", f"turn_{i}", geo,
                    "consciousness research experiment results",
                )

            traj = store.get_trajectory("consciousness", days=1)
            assert traj["total_readings"] == 5
            assert traj["drift"] is not None
            assert traj["drift"]["rank_trend"] > 0
            assert "stability" in traj["drift"]

    def test_trajectory_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            traj = store.get_trajectory("nonexistent concept")
            assert traj["total_readings"] == 0
            assert traj["drift"] is None

    def test_expert_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_anchored(
                "snap1", "turn_1", SAMPLE_GEOMETRY,
                "MoE expert routing analysis",
                expert_geometries=[
                    {"expert_id": 5, "layer": 12, "router_prob": 0.23,
                     "effective_rank": 45.2, "spectral_entropy": 15.1},
                    {"expert_id": 67, "layer": 12, "router_prob": 0.18,
                     "effective_rank": 72.0, "spectral_entropy": 22.5},
                ],
            )

            profile = store.get_expert_profile(5, layer=12)
            assert profile["activation_count"] == 1
            assert profile["avg_geometry"]["effective_rank"] == pytest.approx(45.2)

    def test_expert_profile_multiple(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            for i in range(3):
                store.record_anchored(
                    f"snap_{i}", f"turn_{i}", SAMPLE_GEOMETRY,
                    f"experiment {i} on emotion detection",
                    expert_geometries=[
                        {"expert_id": 67, "layer": 12,
                         "router_prob": 0.2 + i * 0.05,
                         "effective_rank": 60 + i * 10,
                         "spectral_entropy": 20.0},
                    ],
                )

            profile = store.get_expert_profile(67)
            assert profile["activation_count"] == 3
            assert profile["avg_geometry"]["effective_rank"] == pytest.approx(70.0)
            assert len(profile["concept_hashes"]) >= 1

    def test_expert_profile_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            profile = store.get_expert_profile(999)
            assert profile["activation_count"] == 0

    def test_session_geometry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            for i in range(3):
                store.record_anchored(
                    f"snap_{i}", f"turn_{i}", SAMPLE_GEOMETRY,
                    f"turn {i} of the conversation",
                    session_id="sess_abc",
                )
            store.record_anchored(
                "snap_other", "turn_0", SAMPLE_GEOMETRY,
                "different session entirely",
                session_id="sess_xyz",
            )

            readings = store.get_session_geometry("sess_abc")
            assert len(readings) == 3

    def test_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_geometry("old1", "t1", SAMPLE_GEOMETRY)
            store.record_anchored(
                "new1", "t1", SAMPLE_GEOMETRY,
                "consciousness research",
                session_id="s1",
                expert_geometries=[
                    {"expert_id": 5, "layer": 12, "effective_rank": 50},
                ],
            )

            s = store.stats()
            assert s["total_readings"] == 2
            assert s["anchored_readings"] == 1
            assert s["unique_experts"] == 1
            assert s["sessions"] == 1

    def test_different_concepts_dont_cross(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = self._make_store(tmp)
            store.record_anchored("s1", "t1", SAMPLE_GEOMETRY,
                                  "consciousness and qualia")
            store.record_anchored("s2", "t2", SAMPLE_GEOMETRY,
                                  "USB protocol reverse engineering")

            traj_c = store.get_trajectory("consciousness")
            traj_u = store.get_trajectory("USB protocol")

            if traj_c["total_readings"] > 0 and traj_u["total_readings"] > 0:
                assert traj_c["concept_hash"] != traj_u["concept_hash"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
