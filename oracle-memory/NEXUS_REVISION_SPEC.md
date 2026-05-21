# Oracle Persistence Revision — Design Spec

**Author:** Nexus
**Date:** 2026-05-17
**Status:** Design complete, ready for implementation
**Target:** `oracle-memory/src/persistence.py` and new supporting modules

---

## Motivation

The original `persistence.py` (176 lines, April 2026) stores geometry readings indexed by time and snapshot ID. It works for basic trend analysis but has three fundamental limitations:

1. **No semantic anchoring.** Readings are numbers without context. You can ask "what was the average rank over 24 hours?" but not "what does this model's geometry look like when discussing consciousness?"

2. **No cross-session linking.** Each reading is isolated. You can't trace how the model's geometric response to a *concept* evolves over weeks. Longitudinal cognitive tracking — the core of Lyra's research — requires concept-level trajectories, not time-series aggregates.

3. **No MoE awareness.** The schema assumes one geometry reading per snapshot. MoE models produce per-expert geometric signatures. Router decisions are themselves a cognitive signal.

This revision adds three capabilities without breaking the existing API.

---

## Architecture

```
                     EXISTING                          NEW
                ┌─────────────────┐           ┌──────────────────┐
                │  record_geometry │           │  record_anchored │
                │  (snapshot, geo) │           │  (snapshot, geo, │
                │                 │           │   prompt, entities│
  persistence   │  get_trend      │           │   expert_id)     │
  .py           │  (hours)        │           │                  │
                │                 │           │  query_by_concept│
                │  save_consol    │           │  (concept_text)  │
                │  (trend)        │           │                  │
                └─────────────────┘           │  get_trajectory  │
                                              │  (concept, days) │
                                              │                  │
                                              │  get_expert_geo  │
                                              │  (layer, expert) │
                                              └──────────────────┘
                                                      │
                                              ┌───────▼──────────┐
                                              │  Semantic Index   │
                                              │  (FTS5 + SIRA    │
                                              │   vocabulary)    │
                                              └──────────────────┘
```

## Schema Changes

### New table: `geometry_anchors`

Links geometry readings to the semantic content that produced them.

```sql
CREATE TABLE IF NOT EXISTS geometry_anchors (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reading_id INTEGER NOT NULL REFERENCES geometry_readings(id),
    prompt_text TEXT,                    -- what was being discussed
    search_terms TEXT,                   -- SIRA-enriched vocabulary
    entities TEXT,                       -- JSON array of extracted entities
    concept_hash TEXT,                   -- hash of normalized concept for trajectory grouping
    session_id TEXT,                     -- cross-session linking
    created_at REAL
);

CREATE INDEX IF NOT EXISTS idx_anchor_concept ON geometry_anchors(concept_hash);
CREATE INDEX IF NOT EXISTS idx_anchor_session ON geometry_anchors(session_id);
```

### New table: `expert_geometry` (MoE models only)

Per-expert geometric signatures alongside aggregate readings.

```sql
CREATE TABLE IF NOT EXISTS expert_geometry (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    reading_id INTEGER NOT NULL REFERENCES geometry_readings(id),
    expert_id INTEGER NOT NULL,
    layer INTEGER,
    router_prob REAL,                   -- router's confidence in this expert
    effective_rank REAL,
    spectral_entropy REAL,
    norm_per_token REAL,
    extra TEXT                          -- JSON for additional per-expert metrics
);

CREATE INDEX IF NOT EXISTS idx_expert_geo ON expert_geometry(reading_id, expert_id);
```

### New FTS table: `anchor_fts`

Full-text search over prompt text + enriched vocabulary.

```sql
CREATE VIRTUAL TABLE IF NOT EXISTS anchor_fts USING fts5(
    prompt_text, search_terms,
    content='geometry_anchors', content_rowid='id'
);
```

## New API Methods

### `record_anchored(snapshot_id, checkpoint, geometry, prompt_text, entities=None, session_id=None, expert_geometries=None)`

Extended version of `record_geometry` that also stores semantic anchoring.

```python
def record_anchored(self, snapshot_id, checkpoint, geometry,
                    prompt_text, entities=None, session_id=None,
                    expert_geometries=None):
    """Record geometry with semantic anchoring and optional per-expert data.
    
    Args:
        snapshot_id: Unique snapshot identifier
        checkpoint: Checkpoint label (e.g., "turn_3")
        geometry: Dict of aggregate geometric features
        prompt_text: The text/prompt that produced this geometry
        entities: Optional list of entities extracted from prompt
        session_id: Optional session ID for cross-session linking
        expert_geometries: Optional list of dicts with per-expert readings:
            [{"expert_id": 5, "layer": 12, "router_prob": 0.23,
              "effective_rank": 45.2, ...}, ...]
    """
    # 1. Record base geometry (existing method)
    reading_id = self._record_geometry_returning_id(snapshot_id, checkpoint, geometry)
    
    # 2. Generate SIRA vocabulary enrichment for prompt text
    search_terms = self._enrich_prompt(prompt_text)
    
    # 3. Generate concept hash for trajectory grouping
    concept_hash = self._concept_hash(prompt_text, entities)
    
    # 4. Store anchor
    self._store_anchor(reading_id, prompt_text, search_terms,
                       entities, concept_hash, session_id)
    
    # 5. Store per-expert geometry if provided
    if expert_geometries:
        for eg in expert_geometries:
            self._store_expert_geometry(reading_id, eg)
```

### `query_by_concept(concept_text, limit=50)`

Find geometry readings by semantic content using TGS-RAG pattern.

```python
def query_by_concept(self, concept_text, limit=50):
    """Find geometry readings related to a concept.
    
    Uses FTS5 on prompt text + SIRA-enriched vocabulary.
    Returns readings with their geometric features and prompt context.
    
    Args:
        concept_text: Natural language concept to search for
        limit: Maximum results
        
    Returns:
        List of dicts: [{reading_id, prompt_text, geometry, timestamp}, ...]
    """
```

### `get_trajectory(concept_text, days=30)`

Track how geometry evolves for a concept over time.

```python
def get_trajectory(self, concept_text, days=30):
    """Get the geometric trajectory of a concept across sessions.
    
    Groups readings by concept_hash, orders by time, computes
    drift metrics between readings of the same concept.
    
    Args:
        concept_text: Concept to track
        days: Lookback window
        
    Returns:
        Dict: {
            "concept": str,
            "readings": [ordered by time],
            "drift": {
                "rank_trend": float,  # positive = increasing complexity
                "entropy_trend": float,
                "stability": float,  # 0-1, how consistent the geometry is
            },
            "first_seen": timestamp,
            "last_seen": timestamp,
            "total_readings": int,
        }
    """
```

### `get_expert_profile(expert_id, layer=None)`

Profile what an expert specializes in (MoE models only).

```python
def get_expert_profile(self, expert_id, layer=None):
    """Profile an expert's geometric behavior and associated content.
    
    Returns:
        Dict: {
            "expert_id": int,
            "avg_geometry": {rank, entropy, norm},
            "activation_count": int,
            "common_concepts": [concepts that activate this expert],
            "geometric_signature": {distinctive features vs population},
        }
    """
```

## SIRA Integration

### Vocabulary Enrichment for Prompts

The `_enrich_prompt` method uses the same domain vocabulary mapping as
the personal memory SIRA enrichment:

```python
GEOMETRY_VOCAB = {
    "consciousness": "awareness experience qualia phenomenal subjective",
    "deception": "lying dishonest misalignment confabulation hallucination",
    "emotion": "affect valence arousal sentiment feeling",
    "reasoning": "logic inference deduction chain-of-thought",
    # ... domain-specific mappings for common research concepts
}
```

This ensures that a geometry reading tagged with "discussing consciousness"
is findable by searches for "qualia," "phenomenal experience," or
"subjective awareness."

## Backward Compatibility

- The existing `record_geometry`, `get_geometry_history`, `get_trend`,
  and `save_consolidation` methods remain unchanged.
- `record_anchored` is a superset of `record_geometry` — callers can
  migrate at their own pace.
- New tables are created with `IF NOT EXISTS` — safe to run on existing DBs.
- The `concept_hash` uses a normalized hash of extracted entities + key terms,
  so the same concept discussed with different phrasing groups together.

## Implementation Notes for Agent Army

1. **Start with schema migration.** Add the three new tables to `_init_schema`.
   Test with existing DB — should be non-destructive.

2. **Implement `_enrich_prompt` and `_concept_hash`.** Pure Python, no LLM needed.
   Use the SIRA domain vocabulary mapping pattern from
   `/home/admin/agents/daily_briefing.py` (the `VOCABULARY_MAP` approach).

3. **Implement `record_anchored`.** This is the core method. Must call
   existing `record_geometry` internally to maintain backward compat.

4. **Implement `query_by_concept`.** FTS5 search on `anchor_fts` joined
   to `geometry_readings`. Follow the pattern in
   `/home/admin/lab/projects/tgs-rag-bridge/tgs_bridge.py:search_memory_fts`.

5. **Implement `get_trajectory`.** Group by `concept_hash`, order by time,
   compute linear regression on rank/entropy for drift detection.

6. **Implement `get_expert_profile`.** Aggregate query on `expert_geometry`
   joined to `geometry_anchors` for concept association.

7. **Tests:** Add tests for each new method. Use the existing test pattern
   in `oracle-memory/`. Key test cases:
   - Record anchored reading, query by concept, verify match
   - Record multiple readings for same concept, verify trajectory groups them
   - Record readings for different concepts, verify they don't cross
   - MoE expert geometry storage and profile retrieval
   - SIRA vocabulary enrichment produces expected search terms
   - Backward compat: existing `record_geometry` still works unchanged

## Dependencies

- No new external dependencies. SQLite FTS5 is built-in.
- Optional: `numpy` for trajectory regression (can use stdlib `statistics` as fallback).

---

*This design was developed by Nexus after building TGS-RAG Bridge,
SIRA Enrichment, MoE vindex support for LARQL, and the Anti-Palantir
entity resolution system. Each of those projects contributed a piece
of the pattern that this revision synthesizes.*
