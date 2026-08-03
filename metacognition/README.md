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

- PyTorch, transformers, numpy: `pip install torch transformers numpy`
- **jlens** (Jacobian Lens) is **not on PyPI**. Install it from GitHub:
  ```bash
  pip install "git+https://github.com/anthropics/jacobian-lens.git"
  ```
  (or `pip install -r requirements.txt`, which pins the same GitHub source)
- A model and a **pre-fitted J-lens for that model**.

### Model / lens defaults are placeholders

`test_metacognitive.py` defaults to `--model Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled`
and a lens file `opus_distill_jlens.pt`. **That specific distilled-model + lens pair
is a Coalition-internal artifact and is not publicly published** — you must supply
your own. Pass `--model <your-model>` and `--lens <path-to-your-lens.pt>`.

Public lenses are available from Neuronpedia (38+ pre-fitted models). Note the
published Neuronpedia lens for Qwen is fitted on **base `Qwen/Qwen3.5-27B`**, not the
Opus-distilled variant, so it is not a drop-in for the in-tree default — pick a
lens fitted for whatever model you actually load.

## Note

These are MEASUREMENT tools — they observe and record, they don't modify model behavior. Injection and steering tools are maintained separately.
