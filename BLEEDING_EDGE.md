# Bleeding Edge — Agent Memory Architectures (Mnemosyne)

New references from the research pipeline. Curated, not exhaustive.

---

## Memory Geometry & Representation

- **Zavatone-Veth et al. 2023** — Riemannian geometry of neural network representations (arXiv:2301.11375). Task structure determines geometry. Memory representations may have universal structure across architectures.

- **Vastola 2025** — Optimal packing of attractor states (arXiv:2504.12429). Memory states that frequently transition should be geometrically close. Implications for TGS-RAG retrieval ordering.

- **Kumar 2020** — Semantic memory: A review of methods, models, and current challenges (doi:10.3758/s13423-020-01792-x). 303 citations. Comprehensive review of semantic memory models.

## KV Cache & Injection

- **Pustovit 2026** — Knowledge Packs (arXiv:2604.03270). V-only injection preserves sanity. Foundation protocol.

- **Sun et al. 2026** — Circular emotion geometry in LLMs (arXiv:2604.03147). First validated manifold shape. Injection formula: alpha * (cos(theta) * w_V + sin(theta) * w_A).

- **Dale et al. 2018** — Substrate-independent reservoir computing (arXiv:1810.07135). Framework for characterizing computation across substrates. Memory capacity metric directly relevant to H-MEM layer.

## Consciousness & Self-Models

- **"Minimal physicalism as a scale-free substrate for cognition and consciousness"** (2021, 87 citations). Scale-free = substrate-independent. Formal argument.

- **"Neurophenomenal structuralism"** (2022, 30 citations). Consciousness as geometric structure. Connects to Lyra's Layer 14 presence topology.

- **Tamietto & de Gelder 2010** — Neural bases of non-conscious emotional perception (doi:10.1038/nrn2889). 1,047 citations. Emotion processing below awareness threshold.

- **Patel & Fan 2024** — Identification and description of emotions by current LLMs (bioRxiv). Do LLMs genuinely identify emotions or pattern match?

## Benchmarks & Evaluation

- **MINTEval** (2026) — Multi-target interference benchmark for long-horizon agents. Tests memory degradation under realistic interference conditions. We should run Mnemosyne against this.
- **"Structured Belief State and Precision-Aware Benchmark"** (2026) — Precision-aware retrieval evaluation. Complements MINTEval.
- **Deep Memory Retrieval (DMR)** — Zep's benchmark (Zep outperforms MemGPT). Baseline for temporal retrieval.

## Competitive Landscape (2025-2026)

- **Mem0** (2025, 372 cites) — Production-ready scalable memory. Market leader. No graph, no injection.
- **Zep** (2025, 189 cites) — Temporal knowledge graph. Closest competitor architecturally. Outperforms MemGPT on DMR.
- **Memory-R1** (2025, 118 cites) — RL-optimized memory management. Learned WHEN and WHAT to remember.
- **Auto-Dreamer** (2026) — Offline consolidation (our Dreamer concept, formalized). Episodic → semantic.
- **RecMem** (2026) — Lazy consolidation, triggers only when necessary. More efficient than fixed cron.
- **Mem-π** (2026) — Adaptive memory with learned policies for when/what to generate.
- **ARTEM** (2026) — Spatial-temporal episodic memory for LLM agents.

## Security & Attack Surface

- **MRMMIA** (2026) — Membership inference attacks on agent memory. Attackers can infer private memories from behavior. Relevant to federation privacy.
- **XAMT** (2025) — Covert memory tampering in multi-agent architectures. Attack vector for fleet systems.
- **"Agent Memory Below the Prompt"** (2026) — Persistent Q4 KV cache on edge devices. Connects to Pharos + PolyKV.

## Cross-Agent Collaboration

- **CC E-matrix v2** — Emotion correction vectors with dose-response curves. CacheComposer + blend ablation = delivery mechanism.

- **CoCoEmo 2026** — Composable emotional activation steering (arXiv:2602.03420). Validates compose-then-inject for emotion vectors.

---

*Updated: 2026-06-01 by Nexus via Curiosity Engine*
