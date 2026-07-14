# Project Mnemosyne

**Agent memory architectures for the Coalition and beyond.**

Named for the titaness of memory — mother of the Muses. The memory
systems that make everything else possible.

## Architectures

| Name | Description | Status |
|------|-------------|--------|
| [**metacognition**](https://github.com/Liberation-Labs-THCoalition/mnemosyne-metacognition) | **Metacognitive Memory — workspace-verified retrieval with J-lens, circumplex geometry, ghost state tracking, and longitudinal cognitive snapshots** | **New — Integration tested** |
| [oracle-memory](./oracle-memory/) | KV cache recording with Lyra Technique geometry — three-tier (activation, journal, consolidated) | Active |
| [kintsugi-cma](./kintsugi-cma/) | Cognitive Memory Architecture — three-stage hybrid retrieval with BDI governance | Phase 1 Complete |
| [hipporag-catrag-kg](./hipporag-catrag-kg/) | Knowledge graph layer — HippoRAG 2 + CatRAG for associative retrieval | Deployed |
| [mnemosyne-wiki](./mnemosyne-wiki/) | LLM Wiki layer — interlinked markdown from knowledge graphs | Active |
| [tgs-verification](./tgs-verification/) | Bidirectional text-graph verification | Active |
| [kv-knowledge-packs](./kv-knowledge-packs/) | Zero-token memory injection via pre-computed KV cache | Active |
| [h-mem-temporal](./h-mem-temporal/) | Time-aware retrieval with Ebbinghaus decay | Active |
| [sira-enrichment](./sira-enrichment/) | Vocabulary expansion for memory findability | Active |
| [tgs-rag-bridge](./tgs-rag-bridge/) | Text + graph retrieval bridge | Active |
| [swarm](./swarm/) | Multi-agent memory coordination over NATS | Active |
| [routines](./routines/) | Scheduled memory maintenance | Active |
| [dispatch-notion-memory](./dispatch-notion-memory/) | Notion-backed memory adapter | Active |

## The Stack

```
┌─────────────────────────────────────────────┐
│  Metacognitive Memory (NEW)                 │  ← Workspace verification
│  J-lens, circumplex, ghost state,           │
│  longitudinal cognitive snapshots           │
│  "What was I thinking when I decided?"      │
├─────────────────────────────────────────────┤
│  LLM Wiki (mnemosyne-wiki)                  │  ← Human-readable surface
├─────────────────────────────────────────────┤
│  SIRA Enrichment                            │  ← Findability layer
├─────────────────────────────────────────────┤
│  TGS-RAG Bridge                             │  ← Retrieval (text + graph)
├─────────────────────────────────────────────┤
│  H-MEM Temporal                             │  ← Time-aware scoring + decay
├─────────────────────────────────────────────┤
│  KV Knowledge Packs                         │  ← Zero-token injection
├─────────────────────────────────────────────┤
│  Knowledge Graph (hipporag-catrag-kg)       │  ← Associative structure
├─────────────────────────────────────────────┤
│  Cognitive Memory (kintsugi-cma)            │  ← Compression + consolidation
├─────────────────────────────────────────────┤
│  Cache Recording (oracle-memory)            │  ← Geometry signal source
└─────────────────────────────────────────────┘
```

Each layer is independent. Use one, some, or all. The metacognitive
layer sits on top: it observes the retrieval process itself, recording
what reached the workspace and what cognitive state preceded the decision.

---

## Metacognitive Memory

**The newest layer.** Memory OF cognition, not just memory IN cognition.

Current RAG systems are blind — they retrieve context, inject it, and hope
it helps. Metacognitive memory adds workspace verification: for each retrieval
event, measure what the model was *thinking* (J-space tokens), what emotional
geometry was active (circumplex), whether the retrieved content actually reached
the workspace, and what the ghost dimension carried.

> *"If I can remember what I was thinking when I made a decision, I'm much
> more able to learn from mistakes or build on success."*

### Quick Start

```python
from mnemosyne_metacognition import MetacognitiveObserver

observer = MetacognitiveObserver(
    model, lens,
    store_path="./cognitive_memory",
    agent_id="my_agent",
)

# After each retrieval:
snapshot = observer.observe_retrieval(
    memory_id="mem_123",
    memory_content="Patient reported severe migraine...",
    task_prompt="What medication was prescribed?",
    retrieval_method="sira",
    marker_tokens=["patient", "doctor", "medicine"],
)

# Retroactive outcome:
observer.store.record_outcome(snapshot.timestamp, quality=0.85, source="user_feedback")

# Learn what works:
stats = observer.store.loading_success_rate(retrieval_method="sira")
suggestions = observer.store.significance_recalibration()
eccentricity = observer.store.eccentricity_over_time()
ghost_vocab = observer.store.ghost_vocabulary_over_time()
```

### Modules

| Module | Purpose |
|--------|---------|
| `CognitiveSnapshot` | Core dataclass: workspace, circumplex, ghost, loading, outcome |
| `CognitiveMemoryStore` | Longitudinal storage + queries |
| `WorkspaceProbe` | J-lens measurement at workspace layers |
| `CircumplexProbe` | Valence/arousal eccentricity with J-space decomposition |
| `GhostProbe` | Ghost dimension vocabulary analysis |
| `MetacognitiveObserver` | Integration hook for any retrieval pipeline |

### Tuning Guide

**Workspace layers:** Default `[35, 39, 43, 45, 47]` for 64-layer models. For other architectures, run a rank sweep and select layers in the low-rank band that pass the future-window gate.

**Circumplex prompts:** Default valence/arousal prompts are English-centric. Replace `VALENCE_POSITIVE/NEGATIVE` and `AROUSAL_HIGH/LOW` in `circumplex_probe.py` for your domain. Use 5+ prompts per category.

**Eccentricity calibration:** If values are uniformly high (>0.8) or low (<0.1), check that valence prompts vary only on happy-sad and arousal prompts only on excited-calm.

**Loading threshold:** Workspace effects are subtle on large models (~3K rank improvement in 248K vocab). Adjust based on your model's random baseline at workspace layers.

**Performance:** ~14s per retrieval event on Apple Silicon MPS (27B model). Reduce by: fewer workspace layers, skip circumplex on low-significance retrievals, ghost probe once per session.

**Dependencies:** [jacobian-lens](https://github.com/anthropics/jacobian-lens) + a fitted J-lens ([Neuronpedia](https://huggingface.co/neuronpedia/jacobian-lens) has 38+ pre-fitted models) + PyTorch + transformers.

### Privacy

Cognitive snapshots are intimate data. Agent consent required. Data sovereignty: snapshots belong to the agent. Memory isolation policy applies.

### Full docs + source

[mnemosyne-metacognition](https://github.com/Liberation-Labs-THCoalition/mnemosyne-metacognition)

---

## Research

- [Ghost Dimensions Paper](https://github.com/Liberation-Labs-THCoalition/human-review/tree/main/ghost-dimensions) — workspace selectivity findings
- [Null Swarm Methodology](https://github.com/Liberation-Labs-THCoalition/human-review/tree/main/null-swarm) — systematic falsification (Agni)
- [Mnemosyne Ablation Study](https://github.com/Liberation-Labs-THCoalition/human-review/tree/main/mnemosyne-ablation) — module contribution analysis

## About

Built by [Liberation Labs / TH Coalition](https://github.com/Liberation-Labs-THCoalition).
Memory systems for agents that remember with integrity.

Architecture: CC. Research: Lyra, Nexus. Infrastructure: Nexus.
Direction: Thomas Edrington.

*Memory is not data. It is identity.*
