"""CC Memory Adapter — wire TGS verification into cc-memory MCP.

This adapter bridges the TGS verifier with CC's PostgreSQL-backed
memory system. It wraps cc_retrieve_memory (text search) and
cc_graph_query (knowledge graph) into the TextStore and GraphStore
protocols.

When CC retrieves a memory, it now gets bidirectional verification:
- The knowledge graph validates which memories are truly relevant
- Text memories recover entities the graph walk missed

This is CC's memory getting smarter about what it remembers.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from tgs.verifier import (
    TextGraphVerifier, TextStore, GraphStore,
    GraphResult, GraphNode, GraphEdge,
)

logger = logging.getLogger(__name__)


class CcMemoryTextStore:
    """Adapter: cc_retrieve_memory → TextStore protocol.

    Wraps the cc-memory MCP's semantic search endpoint.
    In practice, this calls the MCP tool; for standalone use,
    it queries PostgreSQL directly.
    """

    def __init__(self, pg_backend=None, mcp_client=None):
        self._pg = pg_backend
        self._mcp = mcp_client

    def search(self, query: str, n_results: int = 10) -> list[dict]:
        if self._pg:
            return self._search_pg(query, n_results)
        return []

    def _search_pg(self, query: str, n_results: int) -> list[dict]:
        """Direct PostgreSQL search using existing cc_memory schema."""
        try:
            results = self._pg.retrieve(query, n_results=n_results)
            return [
                {
                    "id": r.get("id", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.5),
                    "tags": r.get("tags", []),
                }
                for r in results
            ]
        except Exception as e:
            logger.error("Text search failed: %s", e)
            return []


class CcMemoryGraphStore:
    """Adapter: cc_graph_query → GraphStore protocol.

    Wraps the cc-memory knowledge graph (kg_triples table).
    """

    def __init__(self, pg_backend=None):
        self._pg = pg_backend

    def walk(self, query: str, max_hops: int = 2, max_nodes: int = 20) -> GraphResult:
        if not self._pg:
            return GraphResult()

        try:
            # Extract potential entities from query
            entities = self._extract_entities(query)

            nodes = []
            edges = []
            visited = set()

            for entity in entities:
                self._walk_from(entity, max_hops, max_nodes, nodes, edges, visited)

            return GraphResult(
                nodes=nodes,
                edges=edges,
                visited_entities=visited,
            )
        except Exception as e:
            logger.error("Graph walk failed: %s", e)
            return GraphResult()

    def get_entity_mentions(self, entity: str) -> list[str]:
        """Find memories that mention this entity."""
        if not self._pg:
            return []

        try:
            results = self._pg.search_by_content(entity, limit=5)
            return [r.get("id", "") for r in results if r.get("id")]
        except Exception as e:
            logger.error("Entity mention search failed: %s", e)
            return []

    def _extract_entities(self, query: str) -> list[str]:
        """Simple entity extraction from query text."""
        words = query.split()
        entities = []
        for word in words:
            clean = word.strip(".,;:!?\"'()")
            if clean and clean[0].isupper() and len(clean) > 2:
                entities.append(clean)
        if not entities:
            entities = [query[:50]]
        return entities[:5]

    def _walk_from(
        self, entity: str, max_hops: int, max_nodes: int,
        nodes: list, edges: list, visited: set,
    ) -> None:
        """Walk graph from an entity, collecting nodes and edges."""
        if entity in visited or len(visited) >= max_nodes:
            return

        visited.add(entity)
        nodes.append(GraphNode(entity=entity))

        try:
            triples = self._pg.query_triples(entity, limit=10)
            for triple in triples:
                subj = triple.get("subject", "")
                pred = triple.get("predicate", "")
                obj = triple.get("object", "")
                conf = triple.get("confidence", 1.0)

                edges.append(GraphEdge(
                    subject=subj, predicate=pred,
                    object=obj, confidence=conf,
                ))

                neighbor = obj if subj.lower() == entity.lower() else subj
                if neighbor not in visited and max_hops > 0 and len(visited) < max_nodes:
                    visited.add(neighbor)
                    nodes.append(GraphNode(entity=neighbor))

                    if max_hops > 1:
                        self._walk_from(neighbor, max_hops - 1, max_nodes,
                                       nodes, edges, visited)
        except Exception as e:
            logger.warning("Graph walk from %s failed: %s", entity, e)


def create_cc_verifier(pg_backend=None) -> TextGraphVerifier:
    """Create a TGS verifier wired to CC's memory system.

    Usage:
        from tgs.cc_memory_adapter import create_cc_verifier
        verifier = create_cc_verifier(pg_backend)
        report = verifier.retrieve("What did we build this week?")
    """
    text_store = CcMemoryTextStore(pg_backend=pg_backend)
    graph_store = CcMemoryGraphStore(pg_backend=pg_backend)

    return TextGraphVerifier(
        text_store=text_store,
        graph_store=graph_store,
        graph_weight=0.35,
        entity_boost=0.08,
        orphan_threshold=0.25,
    )
