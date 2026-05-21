# H-MEM Temporal — Time-Aware Memory Retrieval with Ebbinghaus Decay

**Temporal consolidation tree with robustness scoring for agent memory.**

Memories decay unless reinforced. Experiment results that haven't been
replicated fade. Contradicted findings drop fast. Current validated work
surfaces naturally. No manual curation required.

Based on [H-MEM](https://arxiv.org/abs/2605.15701) (Yu, Fang et al., 2026).
Adapted for the Mnemosyne agent memory stack.

## The Problem

An agent running experiments accumulates results over weeks. Some early
results get superseded by later work. Some get replicated and confirmed.
Without temporal awareness, retrieval treats a falsified finding from two
weeks ago the same as yesterday's validated result. The agent (or a
researcher querying the system) stumbles over stale data.

## The Solution

Three mechanisms working together:

### 1. Temporal-Semantic Tree

Memories are organized in a tree where each level represents a time window:

```
Level 3:  [──────── Month summary ────────]     ← Long-term memory
Level 2:  [── Week 1 ──] [── Week 2 ──]         ← Medium-term
Level 1:  [D1] [D2] [D3] [D4] [D5] [D6] [D7]   ← Short-term
Level 0:  raw memory events (timestamped leaves)  ← Working memory
```

Within each time window, semantically similar memories are consolidated
upward via LLM summary. The tree builds bottom-up during the dreamer's
periodic consolidation pass.

Query decomposition labels each sub-query SHORT/LONG/MIXED to scope
which tree levels to search — recent experiment results vs. long-term
validated knowledge.

### 2. Ebbinghaus Robustness Decay

Every memory has a robustness score that decays over time unless reinforced:

```
R(m, t) = exp(-(t - r_m) / (τ × (1 + η × ln(1 + n_m))))

Where:
  t     = current time
  r_m   = time of last reinforcement (creation or replication)
  τ     = base decay constant (tune per domain)
  η     = reinforcement scaling factor
  n_m   = reinforcement count (how many times confirmed/replicated)
```

**Reinforcement events:**
- A new experiment replicates the finding → `n_m += 1`, `r_m = now`
- A new experiment contradicts the finding → `n_m = 0`, `r_m` unchanged (rapid decay)
- The finding is cited in a consolidation summary → `n_m += 0.5`

**Effect:** A result replicated 3 times decays slowly (high `n_m`). A
one-off finding from 2 weeks ago with no replication has nearly zero
robustness. A contradicted result drops to floor immediately.

### 3. Four-Factor Retrieval Scoring

Extends TGS-RAG's scoring with temporal and robustness factors:

```
Score = α × Semantic + β × EntityCount + γ × Temporal + δ × Robustness

Where:
  Semantic    = normalized text/embedding similarity (from TGS-RAG)
  EntityCount = graph entity overlap (from TGS-RAG)
  Temporal    = time relevance via IoU of query/memory time intervals
  Robustness  = Ebbinghaus decay score R(m, t)
```

Default weights: `α=0.35, β=0.15, γ=0.20, δ=0.30`

Robustness gets high weight because for experiment data, currency and
validation status matter more than raw similarity.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        H-MEM Temporal Layer                         │
│                                                                     │
│  ┌──────────────────┐  ┌─────────────────┐  ┌──────────────────┐  │
│  │  Temporal Tree    │  │  Robustness     │  │  Query Scoper    │  │
│  │                  │  │  Tracker        │  │                  │  │
│  │ Level 0: leaves  │  │                 │  │ Decomposes query │  │
│  │ Level 1: days    │  │ n_m per memory  │  │ into SHORT/LONG  │  │
│  │ Level 2: weeks   │  │ r_m timestamps  │  │ /MIXED sub-      │  │
│  │ Level 3: months  │  │ Ebbinghaus R()  │  │ queries with     │  │
│  │                  │  │                 │  │ temporal hints   │  │
│  └────────┬─────────┘  └────────┬────────┘  └────────┬─────────┘  │
│           │                     │                     │            │
│           └─────────────────────┼─────────────────────┘            │
│                                 ▼                                   │
│                    ┌─────────────────────┐                          │
│                    │  Temporal Scorer     │                          │
│                    │                     │                          │
│                    │  α·Sem + β·Entity   │                          │
│                    │  + γ·Temporal        │                          │
│                    │  + δ·Robustness      │                          │
│                    └────────┬────────────┘                          │
│                             ▼                                       │
│                     Scored, time-aware                               │
│                     retrieval results                                │
└─────────────────────────────────────────────────────────────────────┘
```

## Integration with Mnemosyne Stack

```
┌─────────────────────────────────────────────┐
│  LLM Wiki (mnemosyne-wiki)                  │  ← Human-readable
├─────────────────────────────────────────────┤
│  SIRA Enrichment                            │  ← Findability (each tree level)
├─────────────────────────────────────────────┤
│  TGS-RAG Bridge                             │  ← Retrieval (text + graph)
├─────────────────────────────────────────────┤
│  ★ H-MEM Temporal ★                        │  ← Time-aware scoring + decay
├─────────────────────────────────────────────┤
│  KV Knowledge Packs                         │  ← Zero-token injection
├─────────────────────────────────────────────┤
│  Knowledge Graph (hipporag-catrag-kg)       │  ← Structure
├─────────────────────────────────────────────┤
│  Cognitive Memory (kintsugi-cma)            │  ← Consolidation
├─────────────────────────────────────────────┤
│  Cache Recording (oracle-memory)            │  ← Geometry signal
└─────────────────────────────────────────────┘
```

### Dreamer Integration

The dreamer's periodic pass (cron) handles three temporal tasks:

1. **Tree consolidation** — cluster new leaves by time window and
   semantic similarity, generate LLM summaries for parent nodes
2. **SIRA enrichment** — vocabulary bridge at each new tree level
   (summaries use different vocabulary than raw results)
3. **Robustness updates** — scan for reinforcement events (new
   experiments confirming/contradicting old findings), update `n_m`
   and `r_m` counters

### SIRA Integration

Vocabulary bridging is critical at tree boundaries. A raw experiment
leaf says "E67 AUROC 0.51 on control set." The week-level summary says
"MoE expert specialization hypothesis falsified." SIRA enriches both
so queries at any abstraction level find relevant results.

## Key Design Decisions

### Why Ebbinghaus over simple timestamp ranking?

Timestamp ranking treats all old results equally. Ebbinghaus decay
rewards results that have been reinforced (replicated, cited,
consolidated). A 3-month-old finding replicated 5 times should rank
higher than yesterday's unreplicated result. Pure recency doesn't
capture that.

### Why not version control / contradiction edges?

H-MEM's original paper merges entities without temporal versioning.
We extend this: when the dreamer detects a contradiction (new result
directly opposes old result), it resets `n_m` to 0 on the old result
AND adds a `contradicted_by` link in HippoRAG. This gives us both
soft decay (Ebbinghaus) and hard contradiction tracking (graph edge).

### Why tree levels and not just flat decay?

Query scoping. A researcher asking "what did we find about emotion
steering last week?" needs SHORT scope — only leaves and day-level
summaries from the past 7 days. A researcher asking "what is the
current consensus on MoE expert specialization?" needs LONG scope —
month-level consolidated summaries with high robustness scores. The
tree structure enables this without separate indexes.

## References

- Yu, J., Fang, Y., Liu, X., & Ma, Y. (2026). "H-MEM: A Novel Memory
  Mechanism for Evolving and Retrieving Agent Memory via a Hybrid
  Structure." arXiv:2605.15701
- Ebbinghaus, H. (1885). "Über das Gedächtnis."

---

*Built by Nexus at Liberation Labs. Memory that knows what it knows —
and when it stopped knowing it.*
