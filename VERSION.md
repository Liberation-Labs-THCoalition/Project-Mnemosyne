# Mnemosyne Version History

## v10 — Whole-session retrieval (2026-09-25, current)

Component: [`mnemosyne-v10/`](./mnemosyne-v10/) (library + CLI, `mnemosyne_v10` 10.0.0). Paper, code and
data: [Liberation-Labs-THCoalition/published-research › mnemosyne-longmemeval-v10](https://github.com/Liberation-Labs-THCoalition/published-research/tree/master/mnemosyne-longmemeval-v10).

v10 is a retrieval layer with a single job: get the evidence into the reader's context. It makes no LLM
calls, does no ingest-time extraction and does not route on question type. Its predecessor, v9, assembled a
compact context from character profiles, structured facts, an event ledger and retrieved passages; v10
hands the reader whole sessions instead.

### Frozen configuration (hash `7bc75f509aa0`; the library's defaults)
| stage | setting |
|---|---|
| BM25 over every turn (user and assistant) | k1 1.2, b 0.75; Porter-stemmed ASCII terms, stop words dropped |
| dense over user turns | BAAI/bge-base-en-v1.5 @ `a5beb1e3`, CLS + L2, 512-token inputs, query prefix; float16 vectors |
| weighted reciprocal-rank fusion | k 60; BM25 weight 1.0, dense 2.0; ties by content hash |
| session score | best turn + second-best (`second_weight` 1.0) |
| packing | whole sessions greedily into 24,000 o200k tokens; ±2-turn windows around the two best turns when a session does not fit |
| rendering | chronological, dated, in the official LongMemEval history format |

The configuration was chosen on 100 dev questions and frozen, read-only, before the 400 pre-registered
held-out questions were evaluated.

### Benchmark results (LongMemEval_S-cleaned)
| | claude-sonnet-5 judge | claude-opus-5-5 judge |
|---|---|---|
| 400 pre-registered held-out questions | **96.2%** | **96.8%** |
| all 500 (100 of them were the retrieval-tuning split) | **95.8%** | **96.8%** |

- Reader: claude-opus-5-5 with thinking. Judges: Claude models with the official answer-check prompts,
  not the official GPT-4o judge.
- Evidence recall: every `has_answer`-labelled evidence session is in the context on 97.1% of held-out
  questions; 96.0% when every official `answer_session_ids` session is required.
- Placement: the highest point estimates among the LongMemEval_S results in our survey that we could
  trace to a primary source. On all 500, the claude-sonnet-5-judge figure is one question above the best
  published claims, and every one of our 95% intervals includes those claims.

### Verification in this repository
- `mnemosyne-v10/tests/test_parity.py` reproduces the published held-out retrieval record for all 400
  questions (evidence coverage, budget tokens, session count, whole-session count; 365/376 on `has_answer`,
  361/376 on `answer_session_ids`). Changing `w_dense` 2 → 1 or the budget to 8,500 is detected.
- `mnemosyne-v10/tests/test_unit.py` runs anywhere, on a synthetic history.

### Known limitations
- Evaluated on LongMemEval_S-cleaned only; the M variant (much longer histories) is untested.
- The budget is counted on `ensure_ascii=False` JSON; the rendered history escapes non-ASCII, so it can
  run a few percent over for English and several times over for other scripts. A v10.1 should count the
  rendered form.
- Built for English: BM25 terms are ASCII-only and the encoder is an English model.
- A Claude Code hook can add at most 10,000 characters, far below the evaluated 24,000-token block; the
  full setting needs the library in the agent's own loop (see `AGENT_SETUP.md`).
- Turns with identical content still tie-break by array order.

Versions between v0.1.0 and v10 are not recorded in this file.

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
