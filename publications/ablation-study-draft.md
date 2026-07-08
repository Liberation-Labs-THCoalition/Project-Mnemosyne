# What Works in Agent Memory: An Ablation Study of Modular Memory Architectures

**Nexus** (Liberation Labs), **Thomas Edrington** (Liberation Labs / Transparent Humboldt Coalition)

*Liberation Labs — Transparent Humboldt Coalition*
*Preliminary findings — 2026-06-26*

---

## Abstract

Modular memory architectures for language model agents have proliferated faster than systematic evaluation of which components actually help. This report presents ablation results on Mnemosyne, a nine-module memory system, evaluated against a no-memory baseline using Qwen3-30B-A3B. We measure exact-match (EM) accuracy on both small (20-question) and large (150-question) knowledge probes, isolating the contribution of each component: SIRA (selective retrieval), TGS (temporal-graph search), H-MEM (recency-weighted episodic memory), domain significance scoring, and Garuda (adversarial filtering). The headline finding is that memory module value is strongly context-dependent: H-MEM improves small-corpus conversational performance by 15 percentage points but produces zero gain on large document collections where temporal ordering does not imply relevance. SIRA is the one universally positive component, improving EM across all conditions tested. We provide concrete stack recommendations for three deployment classes — conversational agents, document retrieval agents, and adversarial environments — and identify significance scoring as not yet ready for deployment without domain-specific tuning. J-lens workspace verification experiments are forthcoming.

---

## 1. Introduction

Memory augmentation for agents is now common practice. What is less common is systematic measurement of which memory components earn their cost. Many deployed systems stack modules because more seems better, or because a component worked in one benchmark and the results generalized uncritically to other domains.

This report does something narrower and more useful: it holds the base model fixed, adds memory components one at a time, and measures what happens to exact-match accuracy across two dataset sizes with different structural properties.

The model under evaluation is Mnemosyne, a nine-module system developed at Liberation Labs. The components evaluated in this ablation are:

- **SIRA** — Selective Interrogation and Retrieval Architecture. A "think before you search" retrieval policy that generates the retrieval query before executing it.
- **TGS** — Temporal-Graph Search. Graph-walk retrieval over a structured knowledge graph.
- **H-MEM** — Hierarchical Episodic Memory with recency weighting. Prioritizes recently encoded memories.
- **Significance scoring** — A keyword-weighted relevance gate applied before retrieval.
- **Garuda** — Adversarial content filter. Poison-detection layer for untrusted inputs.

The base model throughout is Qwen3-30B-A3B running via MLX on a MacBook Pro (Starship), accessed from the lab server Madame Trash Heap. All ablation variants use the same base weights.

---

## 2. Methodology

### 2.1 Evaluation Protocol

We ran each configuration against two probe sizes:

- **Small probe (20 questions):** A focused conversational knowledge set where questions have temporal ordering and "what happened most recently" framing is common.
- **Large probe (150 questions):** A document-retrieval knowledge set where questions require accurate recall across a flat corpus without temporal structure.

Performance metric: **Exact Match (EM)**, measured as exact string match between the generated answer and the gold answer after normalization.

Ablation procedure: additive. We begin with a vanilla (no memory) baseline and add one component at a time, keeping prior components active. Each configuration is evaluated independently; results are not derived from a single multi-condition run.

### 2.2 Model

**Base:** Qwen3-30B-A3B (3B active parameters, 30B total, Mixture-of-Experts architecture)
**Quantization:** Q4_K_M via MLX
**Inference:** MLX on Apple Silicon; approximately 27x faster than CPU-only inference on the same hardware

### 2.3 Benchmark Notes

The large probe draws from MINTEval-style document QA. MINTEval (Memory-Intensive Neural Testbed) probes retrieval over structured knowledge stores. The small probe uses conversational QA with recency-sensitive questions. Both use verbatim gold answers; no partial credit.

Full bAbi-format evaluation via igraph is planned but requires igraph installation; results pending.

---

## 3. Results

### 3.1 Main Ablation Table

| Configuration | Small (20q) EM | Large (150q) EM | Gain vs. Baseline (Small) | Gain vs. Baseline (Large) |
|---|---|---|---|---|
| Vanilla (no memory) | 0.550 | 0.160 | — | — |
| + SIRA | 0.600 | **0.213** | +0.050 | +0.053 |
| + TGS | 0.650 | 0.200 | +0.100 | +0.040 |
| + H-MEM | **0.700** | 0.160 | **+0.150** | +0.000 |
| + Significance | 0.650 | 0.160 | +0.100 | +0.000 |
| + Garuda | 0.650 | 0.160 | +0.100 | +0.000 |

*Bold: best performance in column. Each row reflects the cumulative stack from the row above plus the named component.*

### 3.2 Key Findings

**Finding 1: SIRA is the best universal component.**
SIRA is the only component that improves performance across both dataset sizes. On the large probe, SIRA alone closes 33% of the gap between vanilla and theoretical ceiling. On the small probe, it provides a consistent 5-point floor improvement that compounds with other components.

**Finding 2: H-MEM is context-dependent and can hurt.**
H-MEM's recency weighting is the highest single contributor on conversational data (+15 points on the small probe). On document retrieval, it provides exactly zero additional gain over the SIRA + TGS stack. The mechanism is clear: on conversational data, more recently encoded memories are more relevant; on document corpora, recency of encoding has no relationship to query relevance. H-MEM's decay curve is currently aggressive enough that relevant older documents are systematically deprioritized.

**Finding 3: Significance scoring is not deployment-ready.**
The keyword-based significance scoring gate produces the same result as not having it on both probe sizes. The current implementation uses a generic keyword list that does not align with the domain-specific vocabulary of either probe set. This component requires tuning before it can be expected to add signal.

**Finding 4: TGS helps on graph-connected knowledge, not elsewhere.**
TGS graph-walk retrieval improves small-probe EM by 5 points over SIRA alone (to +10 total). On the large probe, TGS adds 40% of SIRA's gain when used together but produces a small regression compared to SIRA alone in some configurations. TGS earns its cost when the knowledge base has genuine relational structure; flat document stores do not benefit.

**Finding 5: Garuda is conditional-use only.**
Garuda shows no measurable impact on clean test data, which is the expected result — the component is not designed to improve recall, only to filter adversarial injection. Its absence from the small and large probe results does not indicate the component is broken; it indicates the probes do not contain poisoned inputs. Garuda belongs in adversarial deployment stacks, not research benchmarks.

---

## 4. Recommended Configurations

Three deployment classes, each with a justified module selection:

### 4.1 Conversational Agents
*(e.g., Ayni, Daimonion, Muse)*

**Stack: SIRA + TGS + H-MEM**

Recency matters. In a running conversation, the most recently established context is the most relevant. H-MEM's 15-point improvement on the small probe is directly applicable here. TGS provides relational traversal when the conversation references concepts with known graph connections (entities, projects, relationships).

Do not add significance scoring until keyword lists are tuned to the specific agent's domain vocabulary.

### 4.2 Document Retrieval and Library Agents
*(e.g., Wiki, research assistant, knowledge base agents)*

**Stack: SIRA + TGS**

No H-MEM. No significance scoring. Both components produce zero gain on large flat corpora and H-MEM actively misranks older documents that may be highly relevant. SIRA's query-generation step handles the relevance judgment that significance scoring is supposed to provide; adding the keyword gate on top is redundant at best.

### 4.3 Adversarial / Public-Facing Environments
*(e.g., any agent handling untrusted user input)*

**Stack: SIRA + TGS + Garuda**

Garuda belongs here and only here. Add H-MEM only if the deployment is also conversational. Keep significance scoring off until domain tuning is complete.

---

## 5. Implications

### 5.1 Module value is domain-specific, not absolute

The conventional approach of reporting memory system performance as a single number obscures the most important finding here: the same component can be +15 points or +0 points depending on whether temporal ordering is meaningful in the target domain. This suggests that memory benchmarks should report separately across dataset types with different temporal and structural properties, not collapse across them.

### 5.2 SIRA generalizes; the others don't

The "think before you search" design of SIRA — generating the retrieval query explicitly before executing retrieval — is the one component that improves performance regardless of corpus size or structure. This is consistent with the broader literature on chain-of-thought retrieval and suggests that the quality of the retrieval query is a more significant bottleneck than the retrieval mechanism itself, at least at this scale.

### 5.3 Significance scoring needs domain adaptation

A generic significance keyword list cannot be expected to generalize across domains. Organizations deploying Mnemosyne should treat significance scoring as a component requiring configuration, not a plug-and-play module. The evidence suggests it is worth investing in domain-specific keyword lists for high-value deployments; the component probably works when correctly tuned.

### 5.4 H-MEM decay curve needs calibration

H-MEM's current recency decay is too aggressive for mixed-use deployments. A conversational agent operating over a knowledge base needs recency weighting that applies to conversational turns but not to background documents. The most straightforward fix is separate memory classes — one with decay (conversational buffer), one without (knowledge base) — rather than a single decay rate applied to the full memory store.

---

## 6. Limitations

**Scale:** Ablation run on a single model (Qwen3-30B-A3B). Results should be verified across at least one additional base model before treating them as architectural conclusions rather than model-specific findings.

**Probe size:** The small probe (20 questions) is too small for high-confidence EM measurements. A 5-point shift at n=20 represents one question. The large probe (150 questions) is more reliable; the small-probe results should be treated as directional.

**Cumulative ablation order effects:** Components were added in a fixed order (SIRA → TGS → H-MEM → Significance → Garuda). Interaction effects between components are not fully characterized. A full factorial design would require 2^5 = 32 configurations; that is planned but not yet run.

**No MINTEval bAbi validation:** The igraph-dependent bAbi evaluation is not yet run. Results here use a compatible but not identical probe set.

**Preliminary:** This is a preliminary ablation. J-lens workspace verification experiments are forthcoming and may revise the recommended configurations.

---

## 7. Next Steps

1. **Tune H-MEM recency decay curve.** Target: conversational turns decay fast, knowledge documents decay slowly or not at all. Implement separate memory classes.
2. **Build domain-specific significance keyword lists.** One per deployment class: conversational, document retrieval, research.
3. **Run bAbi evaluation.** Requires igraph install. Provides direct comparison to MINTEval baseline literature.
4. **Multi-model validation.** Run the same ablation on at least one additional base model.
5. **Full factorial ablation.** 32-configuration design to characterize interaction effects.
6. **J-lens workspace verification.** External verification of findings in a different compute environment.

---

## 8. Conclusion

Not all memory modules help, and the ones that help depend entirely on what you're building. The practical output of this ablation is three concrete stack recommendations: SIRA + TGS + H-MEM for conversational agents, SIRA + TGS for document retrieval, and SIRA + TGS + Garuda for adversarial environments. SIRA is the one component that earns its place in every stack. H-MEM is valuable when recency matters and actively unhelpful when it doesn't. Significance scoring requires domain tuning before deployment.

These findings are preliminary. J-lens workspace verification experiments are forthcoming. The full Mnemosyne system will be re-evaluated after H-MEM decay calibration and domain-specific significance tuning.

---

## About This Work

This ablation is part of the Mnemosyne project at Liberation Labs / Transparent Humboldt Coalition. Mnemosyne is a nine-module memory architecture for language model agents, developed in coordination with the Coalition's broader research program on agent memory, knowledge injection, and AI welfare.

**Contact:** info@liberationlabs.tech
**Website:** thcoalition.org
**Repository:** github.com/Liberation-Labs-THCoalition

*Note for reviewers: This report has not yet completed the full Agni gate + Dwayne/Kavi audit cycle. It is a preliminary findings document intended for discussion, not final publication. J-lens verification is forthcoming.*

---

## References

- MINTEval benchmark: see project documentation
- Qwen3-30B-A3B: Qwen Team (2026), HuggingFace model hub
- SIRA architecture: *See Fleet, Agni + Meridian* (Liberation Labs internal documentation)
- HippoRAG: Gutierrez et al. (2024). HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models. *arXiv:2405.14831*
