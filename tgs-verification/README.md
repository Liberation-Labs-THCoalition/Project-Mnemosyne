# TGS Verification — Bidirectional Text-Graph Memory Verification

Based on [TGS-RAG (2025)](https://arxiv.org/abs/2605.05643), adapted for
agent memory systems.

The problem: text retrieval and graph retrieval operate as independent silos.
Text retrieval returns semantically similar but potentially irrelevant memories.
Graph retrieval prunes paths that might have been valid.

The solution: bidirectional verification.
- **Graph → Text**: Graph nodes vote on text relevance, filtering noise
- **Text → Graph**: Text context resurrects pruned graph paths

## Architecture

```
Query
  ↓
├─ Text Retrieval ──→ candidate memories (semantic search)
├─ Graph Walk ──────→ related entities + paths (knowledge graph)
  ↓
Bidirectional Verification
  ├─ Graph votes on text: re-rank memories by entity coverage
  ├─ Text bridges orphans: recover pruned entities via text context
  ↓
Verified + Completed memory context
```

## Usage

```python
from tgs import TextGraphVerifier, MemoryStore

store = MemoryStore()  # your text + graph backend
verifier = TextGraphVerifier(store)

results = verifier.retrieve("What did we discuss about the Oracle Loop?")
# Returns: verified memories with graph-validated relevance scores
# + recovered connections the graph alone would have missed
```

## Integration Points

- **Mnemosyne stack**: sits between retrieval and the LLM context window
- **cc-memory**: wires into cc_retrieve_memory and cc_graph_query
- **Any RAG system**: generic interface for text store + graph store
