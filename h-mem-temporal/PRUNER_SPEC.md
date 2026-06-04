# The Pruner — Graph-Consistent Controlled Forgetting

**Status:** Spec. Not built.

## The Problem

H-MEM decays memory *records* via Ebbinghaus curves. When a memory's robustness drops below `forget_threshold` (0.1), it gets archived. But the knowledge graph triples that memory created stay in HippoRAG as orphaned connections — ghost edges from forgotten experiences. The graph accumulates noise proportional to its forgetting rate.

## The Fix

The Pruner sits between H-MEM (decides what to forget) and HippoRAG (stores the graph). When a memory is forgotten, the Pruner evaluates its triples and either prunes or downweights them based on support from surviving memories.

## Architecture

```
H-MEM decay sweep
    ↓ (memory below forget_threshold)
Pruner
    ├── 1. Extract triples anchored to the forgotten memory
    ├── 2. For each triple: count supporting memories (other memories that generated the same or equivalent triple)
    ├── 3. Unsupported triples (support_count == 0) → delete via HippoRAG cascade
    ├── 4. Weakly supported triples (support_count == 1) → flag for review at next consolidation
    └── 5. Well-supported triples (support_count >= 2) → keep, remove the forgotten memory's anchor
         ↓
HippoRAG graph (clean, consistent)
```

## Key Design Decisions

### Triple equivalence

Two triples are "equivalent" if they describe the same relationship, not just identical strings. Options:
- **Exact match:** `(s, p, o)` string equality. Fast, misses paraphrases.
- **Embedding similarity:** encode triples, cosine threshold. Catches paraphrases, more expensive.
- **Entity-anchored:** same subject + same object + predicate similarity above threshold. Middle ground.

Start with entity-anchored. SIRA vocabulary expansion on predicates handles synonym predicates ("argues_for" ≈ "supports" ≈ "advocates").

### Cascade vs. soft delete

HippoRAG's `delete()` cascades — removes the document and its graph entities/edges. But we don't want to remove entities that appear in other documents. The Pruner should:
1. Remove the *document* (the memory text)
2. For each entity in that document: check if other documents reference it
3. Only remove entities that become orphaned
4. Remove edges that lose both endpoints

This is a reference-counted cascade, not a blind delete.

### Consolidation timing

The Pruner runs during the Dreamer's consolidation sweep (every 4h, or RecMem-style lazy trigger when implemented). It does NOT run on every memory access — that would make forgetting synchronous and expensive.

Sequence:
1. Dreamer enrichment sweep runs
2. H-MEM decay scoring identifies memories below threshold
3. Pruner evaluates those memories' graph triples
4. Pruner executes deletions/downweights
5. Graph consistency check (optional: verify no dangling edges)

### Metrics

- **Orphan rate:** fraction of graph edges anchored to only one memory. Should decrease after pruning.
- **Graph density before/after:** pruning should reduce noise without destroying connectivity.
- **Retrieval quality:** SIRA-enriched queries should return more relevant results after pruning (less noise).
- **MINTEval scores:** if the benchmark tests interference, pruning should reduce it.

## Interface

```python
class Pruner:
    def __init__(self, hipporag_url: str, hmem_db: str):
        """Connect to HippoRAG and H-MEM's temporal tree."""

    def evaluate(self, memory_id: str) -> PruneDecision:
        """Analyze a forgotten memory's triples for orphan status."""

    def execute(self, decision: PruneDecision) -> PruneResult:
        """Apply the pruning decision to the graph."""

    def sweep(self, forgotten_ids: list[str]) -> SweepResult:
        """Batch prune all newly-forgotten memories."""

    def audit(self) -> AuditResult:
        """Report orphan rate, graph density, dangling edges."""
```

## Dependencies

- HippoRAG API: `delete()`, `query()`, entity/edge listing
- H-MEM temporal_tree: `ebbinghaus_decay()`, forget threshold
- SIRA vocabulary: predicate synonym expansion for equivalence matching

## What This Enables

- **Clean federation:** when memories are pruned locally, their triples don't leak to the federated graph on the next bridge sync
- **Bounded graph growth:** the graph stays proportional to *active* knowledge, not cumulative history
- **Better retrieval:** less noise in the graph means SIRA + TGS-RAG return more relevant results
- **Temporal coherence:** old, forgotten facts don't compete with current knowledge in retrieval

## Connection to SOTA

- RecMem's lazy consolidation trigger would make the Pruner more efficient (don't sweep on cron, sweep when memory pressure builds)
- Zep's temporal edges would give the Pruner time-aware support counting (a triple supported only by old memories is weaker than one supported by recent memories)
- MINTEval's interference benchmark directly tests what the Pruner should fix

---

*The graph remembers what the agent forgets. The Pruner fixes that.*
