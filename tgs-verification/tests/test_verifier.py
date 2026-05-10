"""Tests for bidirectional text-graph verification."""
from __future__ import annotations

from tgs.verifier import (
    TextGraphVerifier, GraphResult, GraphNode, GraphEdge,
    VerifiedMemory, VerificationReport,
)


class MockTextStore:
    def __init__(self, memories: list[dict]):
        self._memories = memories

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        return self._memories[:n_results]


class MockGraphStore:
    def __init__(self, nodes: list[GraphNode], edges: list[GraphEdge], mentions: dict[str, list[str]] = None):
        self._nodes = nodes
        self._edges = edges
        self._mentions = mentions or {}

    def walk(self, query: str, max_hops: int = 2, max_nodes: int = 20) -> GraphResult:
        return GraphResult(
            nodes=self._nodes,
            edges=self._edges,
            visited_entities={n.entity for n in self._nodes},
        )

    def get_entity_mentions(self, entity: str) -> list[str]:
        return self._mentions.get(entity, [])


def _make_stores():
    text_store = MockTextStore([
        {"id": "m1", "content": "The Oracle Loop detects confabulation using KV cache geometry.", "score": 0.9},
        {"id": "m2", "content": "Lyra designed the honesty signal experiment with frequency controls.", "score": 0.85},
        {"id": "m3", "content": "Random unrelated memory about gardening and tomatoes.", "score": 0.7},
        {"id": "m4", "content": "Thomas mentioned the Hand project hackathon at GitHub HQ.", "score": 0.6},
        {"id": "m5", "content": "Vera painted golden roots for Project Raíz.", "score": 0.5},
    ])

    graph_store = MockGraphStore(
        nodes=[
            GraphNode("Oracle Loop", "system"),
            GraphNode("KV cache", "concept"),
            GraphNode("confabulation", "phenomenon"),
            GraphNode("Lyra", "agent"),
        ],
        edges=[
            GraphEdge("Oracle Loop", "uses", "KV cache"),
            GraphEdge("Oracle Loop", "detects", "confabulation"),
            GraphEdge("Lyra", "researched", "KV cache"),
        ],
        mentions={"Thomas": ["m4"], "Vera": ["m5"], "Raíz": ["m5"]},
    )

    return text_store, graph_store


class TestGraphVoting:
    def test_entity_overlap_boosts_score(self):
        text_store, graph_store = _make_stores()
        verifier = TextGraphVerifier(text_store, graph_store)
        report = verifier.retrieve("Oracle Loop confabulation detection")

        # m1 mentions Oracle Loop, confabulation, KV cache — should be top
        top = report.verified_memories[0]
        assert top.memory_id == "m1"
        assert len(top.entity_overlap) >= 2
        assert top.verification == "confirmed"

    def test_no_overlap_weakens(self):
        text_store, graph_store = _make_stores()
        verifier = TextGraphVerifier(text_store, graph_store)
        report = verifier.retrieve("Oracle Loop")

        gardening = [vm for vm in report.verified_memories if "gardening" in vm.content]
        assert len(gardening) == 1
        assert gardening[0].verification == "weakened"

    def test_combined_score_uses_both(self):
        text_store, graph_store = _make_stores()
        verifier = TextGraphVerifier(text_store, graph_store, graph_weight=0.5)
        report = verifier.retrieve("Oracle Loop")

        for vm in report.verified_memories:
            assert vm.combined_score > 0
            assert vm.text_score >= 0
            assert vm.graph_score >= 0

    def test_lyra_mention_gets_partial(self):
        text_store, graph_store = _make_stores()
        verifier = TextGraphVerifier(text_store, graph_store)
        report = verifier.retrieve("KV cache experiment")

        lyra_mem = [vm for vm in report.verified_memories if "Lyra" in vm.content]
        assert len(lyra_mem) == 1
        assert lyra_mem[0].verification in ("confirmed", "partial")


class TestOrphanBridging:
    def test_bridges_orphan_entities(self):
        text_store, graph_store = _make_stores()
        verifier = TextGraphVerifier(text_store, graph_store)
        report = verifier.retrieve("Oracle research")

        # Thomas and Vera are in text but not in graph walk
        assert report.orphan_entities_found > 0

    def test_bridged_entities_boost_score(self):
        text_store, graph_store = _make_stores()
        verifier = TextGraphVerifier(text_store, graph_store, orphan_threshold=0.3)
        report = verifier.retrieve("Oracle research")

        bridged = [vm for vm in report.verified_memories if vm.bridged_entities]
        # Thomas and Vera should be bridged via get_entity_mentions
        if bridged:
            assert bridged[0].combined_score > 0


class TestVerificationReport:
    def test_report_summary(self):
        text_store, graph_store = _make_stores()
        verifier = TextGraphVerifier(text_store, graph_store)
        report = verifier.retrieve("Oracle Loop")

        summary = report.summarize()
        assert "TGS Verification" in summary
        assert "Text candidates" in summary
        assert "Graph nodes" in summary

    def test_respects_n_results(self):
        text_store, graph_store = _make_stores()
        verifier = TextGraphVerifier(text_store, graph_store)
        report = verifier.retrieve("Oracle", n_results=3)
        assert len(report.verified_memories) <= 3

    def test_empty_graph(self):
        text_store = MockTextStore([
            {"id": "m1", "content": "Some memory.", "score": 0.8},
        ])
        graph_store = MockGraphStore([], [])
        verifier = TextGraphVerifier(text_store, graph_store)
        report = verifier.retrieve("anything")
        assert len(report.verified_memories) >= 1
        assert report.verified_memories[0].verification == "text_only"

    def test_empty_text(self):
        text_store = MockTextStore([])
        graph_store = MockGraphStore(
            [GraphNode("test", "concept")],
            [],
        )
        verifier = TextGraphVerifier(text_store, graph_store)
        report = verifier.retrieve("test")
        assert len(report.verified_memories) == 0
