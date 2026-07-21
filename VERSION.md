# Mnemosyne Version History

## v0.1.0 — Baseline (2026-07-21)

First benchmarked version. LoCoMo F1: 0.427 (TF-IDF retrieval, no HippoRAG graph traversal).

### Modules
- SIRA: semantic indexed retrieval with enrichment
- HippoRAG: knowledge graph entity linking (not yet used in benchmark retrieval)
- H-MEM: temporal memory hierarchy
- Significance scoring: auto-score interactions for persistence
- Dreamer: periodic cross-referencing consolidation (every 4h)
- Metacognitive probes: workspace, circumplex, ghost (measurement-only)
- TGS-RAG bridge: connects memory to knowledge graph
- TGS verification: memory consistency checks
- Garuda: poison tasting for input safety

### Benchmark Results (v0.1.0)
| Benchmark | Score | Notes |
|-----------|-------|-------|
| LoCoMo | 0.427 F1 | TF-IDF baseline, no graph retrieval |
| — Adversarial | 0.886 | Strong false-premise rejection |
| — World knowledge | 0.414 | TF-IDF finds relevant context |
| — Temporal | 0.168 | Needs date extraction |
| — Single-hop | 0.160 | Needs HippoRAG graph traversal |
| — Open domain | 0.066 | Needs inference beyond context |

### Known Gaps
- No embedding-based retrieval (TF-IDF only)
- HippoRAG graph not used for retrieval (only storage)
- No consolidation gating (Dreamer can degrade below no-memory baseline)
- No abstention calibration (model guesses when it should say "I don't know")
- Keyword-based evaluation confound identified in ethics pack paper

## Upgrade Path to v0.2.0
1. HippoRAG 2 (passage nodes, unified representations)
2. Embedding retrieval alongside TF-IDF
3. Consolidation gating (novelty gate, raw episode preservation)
4. Abstention confidence threshold
5. Temporal boost for current-state queries
