"""Tests for H-MEM Temporal — tree, scoring, decay, and consolidation."""

import math
import tempfile
import time
from unittest.mock import patch

import pytest

from temporal_tree import (
    TemporalTree, TreeNode, TimeScope,
    ebbinghaus_decay, temporal_iou,
    DAY, WEEK, MONTH,
)
from temporal_scorer import (
    TemporalScorer, ScoringWeights, QueryScoper, ScoredResult,
)
from dreamer_consolidator import (
    DreamerConsolidator, cosine_similarity_text,
)


class TestEbbinghausDecay:
    def test_no_elapsed_time(self):
        now = time.time()
        assert ebbinghaus_decay(now, now, 0) == 1.0

    def test_decays_over_time(self):
        now = time.time()
        r1 = ebbinghaus_decay(now, now - DAY, 0)
        r7 = ebbinghaus_decay(now, now - WEEK, 0)
        r30 = ebbinghaus_decay(now, now - MONTH, 0)
        assert r1 > r7 > r30
        assert r1 < 1.0
        assert r30 > 0.0

    def test_reinforcement_slows_decay(self):
        now = time.time()
        r_unreinforced = ebbinghaus_decay(now, now - WEEK, 0)
        r_reinforced_3 = ebbinghaus_decay(now, now - WEEK, 3)
        r_reinforced_10 = ebbinghaus_decay(now, now - WEEK, 10)
        assert r_reinforced_3 > r_unreinforced
        assert r_reinforced_10 > r_reinforced_3

    def test_zero_reinforcement_decays_fastest(self):
        now = time.time()
        r0 = ebbinghaus_decay(now, now - 2 * WEEK, 0)
        r5 = ebbinghaus_decay(now, now - 2 * WEEK, 5)
        assert r0 < 0.2
        assert r5 > r0


class TestTemporalIoU:
    def test_perfect_overlap(self):
        score = temporal_iou(100, 200, 100, 200)
        assert score == pytest.approx(1.0)

    def test_no_overlap(self):
        score = temporal_iou(100, 200, 300, 400)
        assert score < 0.5

    def test_partial_overlap(self):
        score = temporal_iou(100, 200, 150, 250)
        assert 0.3 < score < 0.8

    def test_contained(self):
        score = temporal_iou(100, 400, 200, 300)
        assert score > 0.3


class TestTemporalTree:
    def test_add_and_get_leaf(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            tree = TemporalTree(f.name)
            nid = tree.add_leaf("test memory", metadata={"source": "test"})
            node = tree.get_node(nid)
            assert node.content == "test memory"
            assert node.level == 0
            assert node.reinforcement_count == 0
            tree.close()

    def test_reinforce(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            tree = TemporalTree(f.name)
            nid = tree.add_leaf("experiment result A")
            tree.reinforce(nid)
            tree.reinforce(nid)
            node = tree.get_node(nid)
            assert node.reinforcement_count == 2
            assert node.last_reinforced > 0
            tree.close()

    def test_contradict(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            tree = TemporalTree(f.name)
            old = tree.add_leaf("E67 is an emotion hub")
            new = tree.add_leaf("E67 routes all concepts equally")
            tree.reinforce(old, 5)
            tree.contradict(old, new)
            node = tree.get_node(old)
            assert node.reinforcement_count == 0
            assert node.contradicted_by == new
            tree.close()

    def test_create_parent(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            tree = TemporalTree(f.name)
            c1 = tree.add_leaf("finding A", timestamp=time.time() - 100)
            c2 = tree.add_leaf("finding B", timestamp=time.time() - 50)
            children = [tree.get_node(c1), tree.get_node(c2)]

            pid = tree.create_parent(children, "Summary of A and B", level=1)
            parent = tree.get_node(pid)
            assert parent.level == 1
            assert len(parent.children_ids) == 2

            child = tree.get_node(c1)
            assert child.parent_id == pid
            tree.close()

    def test_unconsolidated_leaves(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            tree = TemporalTree(f.name)
            tree.add_leaf("orphan 1")
            tree.add_leaf("orphan 2")
            c3 = tree.add_leaf("consolidated")
            tree.create_parent([tree.get_node(c3)], "parent", level=1)

            orphans = tree.get_unconsolidated_leaves()
            assert len(orphans) == 2
            tree.close()

    def test_search_scoping(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            tree = TemporalTree(f.name)
            now = time.time()
            tree.add_leaf("recent", timestamp=now)
            tree.add_leaf("old", timestamp=now - MONTH)

            c1 = tree.add_leaf("for parent", timestamp=now - WEEK)
            tree.create_parent([tree.get_node(c1)], "week summary", level=2)

            short = tree.search(TimeScope.SHORT, time_hint=(now - DAY, now))
            assert len(short) == 1
            assert short[0].content == "recent"

            long_results = tree.search(TimeScope.LONG)
            assert any(r.level >= 2 for r in long_results)
            tree.close()

    def test_stats(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            tree = TemporalTree(f.name)
            tree.add_leaf("a")
            tree.add_leaf("b")
            s = tree.stats()
            assert s["total_nodes"] == 2
            assert s["per_level"][0] == 2
            tree.close()


class TestTemporalScorer:
    def test_contradicted_results_penalized(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            tree = TemporalTree(f.name)
            now = time.time()
            old = tree.add_leaf("E67 is emotion hub", timestamp=now - WEEK)
            new = tree.add_leaf("E67 routes everything", timestamp=now)
            tree.contradict(old, new)

            scorer = TemporalScorer(tree)
            results = scorer.score_results([
                {"content": "E67 is emotion hub", "tgs_score": 0.9, "id": old, "rec_count": 5},
                {"content": "E67 routes everything", "tgs_score": 0.8, "id": new, "rec_count": 3},
            ], now=now)

            contradicted = next(r for r in results if r.contradicted)
            valid = next(r for r in results if not r.contradicted)
            assert valid.combined_score > contradicted.combined_score
            tree.close()

    def test_reinforced_ranks_higher(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            tree = TemporalTree(f.name)
            now = time.time()
            strong = tree.add_leaf("replicated finding", timestamp=now - 2 * WEEK)
            tree.reinforce(strong, 5)
            weak = tree.add_leaf("one-off result", timestamp=now - 2 * WEEK)

            scorer = TemporalScorer(tree)
            results = scorer.score_results([
                {"content": "replicated finding", "tgs_score": 0.7, "id": strong, "rec_count": 2},
                {"content": "one-off result", "tgs_score": 0.7, "id": weak, "rec_count": 2},
            ], now=now)

            assert results[0].content == "replicated finding"
            assert results[0].robustness_score > results[1].robustness_score
            tree.close()

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError):
            ScoringWeights(semantic=0.5, entity_count=0.5,
                          temporal=0.5, robustness=0.5).validate()


class TestQueryScoper:
    def test_recent_query(self):
        scoper = QueryScoper()
        scope, hint = scoper.scope_query("what did we find today about MoE?")
        assert scope == TimeScope.SHORT
        assert hint is not None

    def test_historical_query(self):
        scoper = QueryScoper()
        scope, hint = scoper.scope_query("what is the established consensus on attention geometry?")
        assert scope == TimeScope.LONG

    def test_mixed_query(self):
        scoper = QueryScoper()
        scope, hint = scoper.scope_query("how does expert routing work in transformers?")
        assert scope == TimeScope.MIXED


class TestCosineSimilarity:
    def test_identical(self):
        assert cosine_similarity_text("hello world", "hello world") == pytest.approx(1.0)

    def test_no_overlap(self):
        assert cosine_similarity_text("hello world", "foo bar") == 0.0

    def test_partial(self):
        sim = cosine_similarity_text("KV cache geometry research", "cache geometry analysis")
        assert 0.3 < sim < 1.0


class TestDreamerConsolidator:
    def test_cluster_by_similarity(self):
        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            tree = TemporalTree(f.name)
            consolidator = DreamerConsolidator(tree)

            nodes = [
                TreeNode(id=1, content="KV cache experiment A", timestamp=0,
                         level=0, window_start=0, window_end=0),
                TreeNode(id=2, content="KV cache experiment B", timestamp=0,
                         level=0, window_start=0, window_end=0),
                TreeNode(id=3, content="sewing machine USB protocol", timestamp=0,
                         level=0, window_start=0, window_end=0),
            ]

            clusters = consolidator._cluster_by_similarity(nodes, threshold=0.3)
            kv_cluster = [c for c in clusters if len(c) > 1]
            assert len(kv_cluster) == 1
            assert all("KV" in n.content for n in kv_cluster[0])
            tree.close()

    @patch("dreamer_consolidator.llm_generate")
    def test_consolidate_level(self, mock_llm):
        mock_llm.return_value = "Summary: two related KV cache geometry experiments on deception detection."

        with tempfile.NamedTemporaryFile(suffix=".db") as f:
            tree = TemporalTree(f.name)
            now = time.time()
            tree.add_leaf(
                "KV cache geometry experiment deception detection layer 12 attention heads eigenvalue analysis",
                timestamp=now
            )
            tree.add_leaf(
                "KV cache geometry experiment deception detection layer 14 attention heads spectral features",
                timestamp=now + 10
            )
            tree.add_leaf("sewing machine USB protocol reverse engineering", timestamp=now + 20)

            consolidator = DreamerConsolidator(tree)
            result = consolidator.consolidate_level(source_level=0)

            assert result["parents_created"] >= 1
            assert result["consolidated"] >= 2

            s = tree.stats()
            assert 1 in s["per_level"]
            tree.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
