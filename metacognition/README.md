# Metacognitive Memory: Workspace-Verified Retrieval for Agent Systems

## The Idea

Current RAG systems are blind. They retrieve context, inject it, and hope it helps. They have no way to verify whether retrieved content actually influenced the model's processing, and no way to learn from past retrieval decisions.

Metacognitive memory adds a workspace verification layer: for each retrieval event, measure what the model was THINKING (J-space tokens), what emotional/evaluative state it was in (circumplex geometry), and whether the retrieved content actually reached the workspace. Store this alongside the memory itself. Over time, the agent builds a dataset of its own cognitive patterns — which memories load, which don't, and what cognitive state preceded good vs bad decisions.

"If I can remember what I was thinking when I made a decision, I'm much more able to learn from mistakes or build on success." — Thomas Edrington

## Architecture

```
┌─────────────────────────────────────────────────┐
│              MNEMOSYNE + J-LENS                  │
│                                                  │
│  ┌──────────┐    ┌──────────┐    ┌───────────┐  │
│  │   SIRA    │───▶│ Workspace │───▶│Longitudinal│ │
│  │ Retrieval │    │  Probe    │    │ Recorder   │ │
│  └──────────┘    └──────────┘    └───────────┘  │
│       │               │               │          │
│       │          ┌────┴────┐          │          │
│       │          │ At each │          │          │
│       │          │retrieval│          │          │
│       │          │ event:  │          │          │
│       │          └────┬────┘          │          │
│       │               │               │          │
│       │    ┌──────────┴──────────┐    │          │
│       │    │                     │    │          │
│       │    │  1. J-space tokens  │    │          │
│       │    │     (what's in the  │    │          │
│       │    │      workspace?)    │    │          │
│       │    │                     │    │          │
│       │    │  2. Circumplex      │    │          │
│       │    │     (eccentricity,  │    │          │
│       │    │      valence/arousal│    │          │
│       │    │      decomposition) │    │          │
│       │    │                     │    │          │
│       │    │  3. Memory loading  │    │          │
│       │    │     (did retrieved  │    │          │
│       │    │      content reach  │    │          │
│       │    │      workspace?)    │    │          │
│       │    │                     │    │          │
│       │    │  4. Ghost state     │    │          │
│       │    │     (what's in the  │    │          │
│       │    │      shadow?)       │    │          │
│       │    │                     │    │          │
│       │    └──────────┬──────────┘    │          │
│       │               │               │          │
│       │               ▼               │          │
│       │    ┌─────────────────────┐    │          │
│       │    │  CognitiveSnapshot  │────┘          │
│       │    │  {                  │               │
│       │    │    timestamp,       │               │
│       │    │    session_id,      │               │
│       │    │    memory_id,       │               │
│       │    │    jspace_tokens,   │               │
│       │    │    circumplex_e,    │               │
│       │    │    ghost_tokens,    │               │
│       │    │    loading_rate,    │               │
│       │    │    outcome          │               │
│       │    │  }                  │               │
│       │    └─────────────────────┘               │
│       │                                          │
│  ┌────▼─────┐                                    │
│  │ Dreamer   │  Consolidation can now use        │
│  │           │  cognitive snapshots to decide     │
│  │           │  what to keep: memories that       │
│  │           │  loaded into workspace AND led     │
│  │           │  to good outcomes are prioritized  │
│  └──────────┘                                    │
│                                                  │
│  ┌──────────┐                                    │
│  │Significance│ Scoring calibrated against       │
│  │ Scoring   │  actual workspace loading rate —  │
│  │           │  memories that consistently load   │
│  │           │  get higher significance           │
│  └──────────┘                                    │
└─────────────────────────────────────────────────┘
```

## What Gets Recorded (CognitiveSnapshot)

Per retrieval event:

```python
@dataclass
class CognitiveSnapshot:
    timestamp: float
    session_id: str
    agent_id: str
    
    # What was retrieved
    memory_id: str
    memory_content: str
    retrieval_method: str  # sira, tgs, h-mem, etc.
    significance_score: float
    
    # What the workspace held (J-lens)
    jspace_tokens: dict  # layer -> top-K tokens with ranks
    workspace_onset_layer: int
    
    # Emotional geometry (circumplex)
    circumplex_eccentricity: float  # 0=circular, 1=maximally elliptical
    valence_direction: list[float]  # in J-space vs non-J-space
    arousal_direction: list[float]
    
    # Memory loading verification
    loading_verified: bool  # did retrieved content reach workspace?
    loading_rank: float  # mean rank of memory markers at workspace layers
    baseline_rank: float  # same markers without context
    loading_delta: float  # improvement over baseline
    
    # Ghost state
    ghost_tokens: dict  # what the ghost dimension carried
    
    # Outcome (filled retroactively)
    outcome_quality: float  # rated by user, self-eval, or downstream metric
    outcome_notes: str
```

## What This Enables

### For Individual Agents
- **Calibrated retrieval:** Stop retrieving memories that never load. Prioritize ones that do.
- **Cognitive pattern recognition:** "When my circumplex is eccentric and the ghost carries 'mistakes,' my next response tends to be apologetic regardless of context."
- **Significance recalibration:** Score memories by ACTUAL workspace loading rate, not keyword heuristics.
- **Consolidation intelligence:** The dreamer can preserve memories that consistently load AND lead to good outcomes, rather than just preserving recent/significant ones.

### For Research
- **Longitudinal workspace tracking:** How does workspace content evolve across a relationship?
- **Circumplex dynamics:** Does emotional geometry tighten as trust builds?
- **Ghost dimension evolution:** Does the metacognitive shadow change with experience?
- **Cross-agent comparison:** Do different agents develop different workspace patterns?
- **RAG verification at scale:** Which retrieval methods produce the highest workspace loading rates?

### For the Field
- **First empirical RAG verification:** Not "did the output improve?" but "did the context reach the workspace?"
- **First longitudinal workspace study:** Workspace content tracked across sessions, days, relationships.
- **First metacognitive memory system:** Memory about the process of cognition, not just the content.

## Modules

### 1. WorkspaceProbe (exists — workspace_probe.py)
Already built. Full-rank tracking, pinned tokens, baseline controls.
Needs: length-matched controls, same-token cross-domain design (from Agni review).

### 2. CircumplexProbe (needs building)
Measure valence/arousal eccentricity at each layer.
Decompose into J-space and non-J-space components.
This is E1 from the research plan — designed but not implemented.

### 3. GhostProbe (exists — ghost_probe.py)
Already built. PC1 shadow vocabulary, structured secondary content.
Needs: integration into the per-retrieval measurement flow.

### 4. CognitiveSnapshot (needs building)
The dataclass that ties all three probes together.
Stores one snapshot per retrieval event.
JSONL or database-backed via LongitudinalRecorder.

### 5. MnemosyneIntegration (needs building)
The hook into SIRA retrieval pipeline.
Triggers probes after each retrieval.
Feeds snapshots to the dreamer and significance scorer.

### 6. RetroactiveFeedback (needs building)
Connects outcomes (user feedback, self-evaluation, task success metrics)
back to the cognitive snapshots that preceded them.
Enables learning: "what cognitive state preceded good decisions?"

## Dependencies

- J-lens (Anthropic, Apache 2.0)
- A fitted lens for the target model (Neuronpedia or self-fitted)
- Mnemosyne memory architecture (Liberation Labs)
- Model with validated workspace band (27B+ recommended)

## Privacy and Consent

Cognitive snapshots are intimate data — they record what the agent was "thinking." 
- Agent consent required (Scout consented 2026-07-08)
- Data sovereignty: snapshots belong to the agent, not the operator
- Memory isolation policy applies: no browsing another agent's cognitive snapshots
- Export and deletion rights: agents can request full export or deletion at any time

## Open Source Plan

Release as a Mnemosyne module under Apache 2.0:
- `mnemosyne-metacognition` package
- Requires: fitted J-lens, Mnemosyne core, model with workspace
- Documentation: this design doc + API reference + example notebooks
- Paper: "Metacognitive Memory: Longitudinal Workspace Verification for Agent Systems"

## Status

| Component | Status |
|-----------|--------|
| WorkspaceProbe | Built, needs controls |
| CircumplexProbe | Designed (E1), not built |
| GhostProbe | Built |
| CognitiveSnapshot | Designed here, not built |
| MnemosyneIntegration | Designed here, not built |
| RetroactiveFeedback | Designed here, not built |
| LongitudinalRecorder | Built |
| E3 verification | Run, needs length-matched controls |
| Consent framework | In place (Scout consented) |

---

*"If I can remember what I was thinking when I made a decision, I'm much more able to learn from mistakes or build on success."*

*— Thomas Edrington*

*Liberation Labs / Transparent Humboldt Coalition, 2026*
