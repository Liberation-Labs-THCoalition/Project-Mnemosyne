# KV Decomposition + Scale Study — Findings for Paper

## Core Results

### 1. K/V Superadditivity (Qwen2.5-1.5B, toy graph, 21 nodes)

| Condition | Sanity | Multi-hop |
|-----------|--------|-----------|
| BASELINE | 1.000 | 0.167 |
| FULL_KV | 1.000 | **1.000** |
| V_ONLY | 1.000 | 0.333 |
| K_ONLY | 1.000 | 0.333 |
| TEXT_CONTEXT | 1.000 | 0.833 |

- 0.333 + 0.333 < 1.000 — superadditive combination
- FULL_KV > TEXT_CONTEXT — cache injection outperforms text-in-prompt
- Zero sanity degradation across ALL conditions (with deep copy fix)

### 2. DynamicCache Mutation Bug

HuggingFace Transformers DynamicCache is mutated in-place during forward pass.
Without deep copy before each query, previous queries corrupt the cache.
This caused the "Paris cascade" failure and likely the sanity degradation in the powered study.
Fix: clone K/V tensors before reuse.

### 3. Scale Boundary (Qwen2.5-1.5B → Qwen3-30B-A3B)

**1.5B model, real ethics data (361 nodes, 5348 tokens):**

| Condition | Keyword Score | LLM Judge Score |
|-----------|--------------|-----------------|
| BASELINE | 0.000 | 0.000 |
| FULL_KV | 0.000 | 0.050 |
| V_ONLY | 0.000 | 0.100 |
| K_ONLY | 0.000 | 0.050 |
| TEXT_CONTEXT | 0.000 | 0.200 |

**30B MoE, real ethics data (same graph, text-in-prompt):**

| Condition | Keyword Score | Claude Judge Score |
|-----------|--------------|-------------------|
| BASELINE | 0.000 | 0.000 |
| TEXT_SMALL (1 pack, 1313 words) | 0.100 | **0.111** |
| TEXT_LARGE (5 packs, 6895 words) | 0.133 | **0.385** |

Key findings:
- **Scale matters:** 30B outperforms 1.5B on large graphs
- **More context helps at scale:** TEXT_LARGE > TEXT_SMALL on 30B (inverted on 1.5B)
- **Keyword scoring is invalid:** 0.133 keyword → 0.385 judge = 3x undercount
- **Question generation confound:** auto-generated predicates don't always match graph edges

### 4. RoPE Analysis

K vectors at same position from different texts have cosine similarity 0.9999.
RoPE positional encoding dominates; content difference is negligible.
The hybrid cache (K from text A, V from text B) has positionally coherent K vectors.
The scale limitation is NOT positional corruption — it's attention budget.

### 5. Direct Tensor Injection

Three versions tested:
- v1 (random projections): total failure
- v2 (model projections, no residual): repetition loop
- v3 (authentic K/V + graph blend): coherent, content-aware responses

v3 proved that models CAN attend through synthetically constructed cache positions.
But single-position-per-node is too sparse for content retrieval.
Blend ratio ablation (0.0–0.9) showed zero graph awareness at all ratios.
The text-mediated path provides necessary token density.

### 6. Ethics KV Pack Library

30 packs from Stanford Encyclopedia of Philosophy:
- 3,538 nodes, 2,438 edges, 2,506 triples
- ~36,668 walk encoding tokens
- Cross-cultural: African, Buddhist, Chinese, Ancient Greek, feminist, analytic
- Dual purpose: research data + Oracle ethics library

## Scoring Methodology Evolution

1. **Keyword matching** — fast but invalid for natural language entities
2. **DeepSeek v2 LLM judge** — better but still limited by local model capability
3. **Claude subagent judge** — best discernment, used for final scoring
4. **Recommendation:** always use LLM judge for natural-language graph entities

## Paper Structure (Lyra's notes incorporated)

1. Open with superadditivity: 0.333 + 0.333 < 1.000
2. Include SmolLM-135M failure as model-size boundary
3. DynamicCache bug as community service
4. Scale findings: 1.5B vs 30B on real ethics data
5. Question generation confound: honest about predicate mismatch
6. Scoring methodology: keyword → LLM judge evolution
7. Conclude: small focused packs + routing > monolithic injection

## Agni Validation

- Complementarity hypothesis: Gate 1 PASS, Gate 3 PASS
- Injection > text hypothesis: Gate 1 PASS, Gate 3 FAIL (underpowered)
- Ethics pack pipeline: approved with density caveat
- Paper draft: constructive review, no rejections

## Critical Finding: Walk Encoding vs Predicate Questions (2026-05-30)

The walk encoding format encodes TOPOLOGY (connectivity + edge weights) but NOT
SEMANTICS (relationship predicates). Questions asking "what is X connected to
via 'argues_for'?" cannot be answered from walk encoding because the predicate
"argues_for" doesn't appear in the encoding — only weighted connections appear.

Claude judge confirmed: dominant failure mode is model saying "I cannot find
this relationship" — CORRECTLY, because the predicate isn't in the encoding.

TEXT_SMALL (0.050) performed WORSE than BASELINE (0.060) because the walk
encoding causes the model to abandon parametric knowledge without providing
the predicate-level information needed to answer.

**Implication:** Walk encoding evaluation must use topology-appropriate questions:
- "What is X's strongest connection?" (answerable from weights)
- "Is X connected to Y?" (answerable from adjacency)
- "What nodes are in X's neighborhood?" (answerable from walk probabilities)
- NOT "What is X connected to via [predicate]?" (requires triple data, not walks)

The toy graph experiments (21 nodes) succeeded because the questions were
topology-appropriate (bridges, clusters, isolates, reachability). The ethics
graph experiments failed partly because the question generator used predicate-
specific queries against a predicate-free encoding.

**Next step:** Redesign question generator to ask topology questions, or
use the raw triples text (not walk encoding) for predicate-specific questions.

## Encoding Format Comparison — Pharos Results (2026-05-31)

Full 5-condition × 5-theory comparison on Qwen3-30B-A3B (MoReBench, 600s timeout, 0 timeouts):

| Condition | Overall | Delta | Best Use |
|-----------|---------|-------|----------|
| BASELINE | 0.841 | — | — |
| TRIPLES_ONLY | 0.849 | +0.007 | General purpose — consistent, never harmful |
| TRIPLES_SOURCE | 0.848 | +0.006 | High-confidence matched injection |
| WALK_ONLY | 0.823 | -0.019 | Topology queries only — hurts ethical reasoning |
| HYBRID | 0.735 | -0.106 | ONLY with perfect routing — catastrophic when mismatched |

**Key result:** Aristotle pack on Aristotle dilemmas:
- HYBRID: 0.964 (+0.039) — strongest result measured
- TRIPLES_SOURCE: 0.957 (+0.032)
- Triples: 0.931 (+0.006)

**Pharos production architecture:** Triples as default. Source excerpts added when routing confidence is high. Never blind hybrid. Confidence-gated injection.
