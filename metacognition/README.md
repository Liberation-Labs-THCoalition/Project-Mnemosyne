# Metacognitive Memory — Measurement Probes

Read-only measurement layer on top of Mnemosyne retrieval. For each memory retrieval event, records:

1. **Workspace state** — J-lens vocabulary at workspace layers
2. **Emotional geometry** — circumplex eccentricity (valence/arousal balance)
3. **Ghost vocabulary** — what PC1 carries that the workspace excludes
4. **Memory loading** — did the retrieved content reach the workspace?

## Modules

- `cognitive_snapshot.py` — CognitiveSnapshot dataclass + CognitiveMemoryStore
- `circumplex_probe.py` — CircumplexProbe (valence/arousal geometry)
- `workspace_probe.py` — WorkspaceProbe (J-lens workspace layer tracking)
- `ghost_probe.py` — Ghost dimension measurement (logit vs J-lens cosine)
- `mnemosyne_integration.py` — MetacognitiveObserver (hooks probes into retrieval pipeline)
- `test_metacognitive.py` — Integration tests

## Requirements

- A pre-fitted J-lens for your model (available from Neuronpedia for Qwen 3.6 27B and others)
- `pip install jlens torch transformers numpy`

## Note

These are MEASUREMENT tools — they observe and record, they don't modify model behavior. Injection and steering tools are maintained separately.
