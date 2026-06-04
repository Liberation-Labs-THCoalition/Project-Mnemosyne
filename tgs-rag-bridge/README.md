# TGS-RAG Bridge

Text-Graph Synergistic retrieval that fuses knowledge graph traversal with text search through bidirectional verification. Based on [arXiv:2605.05643](https://arxiv.org/abs/2605.05643), adapted for agent memory systems.

## What This Solves

Agent memory systems typically have two retrieval paths that don't talk to each other:
- **Text search** (FTS5, BM25, vector embeddings) — finds memories by content similarity
- **Graph traversal** (knowledge graph, entity-relation triples) — finds memories by structural connections

These paths return different results with different strengths. Text search finds semantically similar content but misses structural relationships. Graph traversal finds connected entities but misses content that uses different vocabulary.

TGS-RAG bridges them: graph structure validates text results, and text content discovers graph paths that were pruned during traversal.

## The Core Algorithm

### Graph → Text: Global Voting

After graph traversal, ALL visited entities (including pruned paths) vote for text chunks:

```
Score_final(chunk) = α × Norm(similarity) + (1-α) × Norm(entity_count)
```

- `similarity` = text search relevance score (FTS rank or cosine similarity)
- `entity_count` = number of graph-visited entities that appear in this chunk
- `α = 0.5` (balances semantic relevance vs structural endorsement)

Chunks endorsed by many graph entities rise in rank even if their raw text similarity is mediocre. This surfaces memories that are structurally relevant but use different vocabulary.

### Text → Graph: Orphan Entity Bridging

After text retrieval, extract entities from top results. Any entity found in text but NOT in the graph traversal is an "orphan." For each orphan:

1. Check if it was explored but pruned during graph beam search
2. If found in the visited-node cache, recover its stored path (zero additional traversal)
3. Score recovered paths: `Score_conf = Score_base + ε × |shared_entities|`

Cap orphan resurrections at `k_o = 3` to prevent noise amplification.

## Architecture

```
┌─────────────────────────────────────────────────┐
│              TGS-RAG Bridge                      │
│                                                  │
│  Query ─┬─→ Graph Backend ──→ Visited Entities   │
│         │   (HippoRAG)         │                 │
│         │                      ▼                 │
│         │              Global Voting             │
│         │                      │                 │
│         └─→ Text Backend ──→ Re-ranked Results   │
│             (SQLite FTS5)      │                 │
│                                ▼                 │
│                        Orphan Bridging           │
│                                │                 │
│                                ▼                 │
│                        Fused Results             │
└─────────────────────────────────────────────────┘
```

## Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.5 | Balance between text similarity and graph endorsement |
| `orphan_cap` | 3 | Maximum orphan entities to bridge per query |
| `epsilon` | 0.4 | Weight for recovered orphan paths |

## Requirements

- A graph retrieval backend (HippoRAG, Neo4j, or any KG with entity search)
- A text retrieval backend (SQLite FTS5, Elasticsearch, or any text search)
- Both backends indexed on the same memory corpus

## Reference Implementation

The production implementation runs on Madame Trash Heap (Liberation Labs) as a systemd service:
- Port 11236
- Bridges HippoRAG (port 11235) and MCP Memory Service (SQLite)
- API: `POST /retrieve` for fused queries, `POST /retrieve/text` and `POST /retrieve/graph` for individual backends

Source: [tgs_bridge.py](./tgs_bridge.py)

## Research Basis

- **TGS-RAG** (arXiv:2605.05643, May 2026): Text-Graph Synergy RAG with bidirectional verification and Global Voting
- **HippoRAG 2** (arXiv:2502.14802, ICML 2025): Neurobiologically-inspired KG + Personalized PageRank

## Author

Nexus — Liberation Labs / Transparent Humboldt Coalition
