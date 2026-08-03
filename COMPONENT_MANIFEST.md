# Component Manifest

Honest status of every component advertised by Project Mnemosyne, so a fresh
clone knows what is runnable now, what is external/gated, and what is still a
design spec. Created in response to issue #4 (reproducibility).

**Legend**
- **Implemented** — source is in this repository and runs/tests from a clean checkout.
- **External / private** — referenced by the docs but **not present in this public repo** (gated, private, or a separate artifact).
- **Design-only** — specification / README in-tree; no runnable service yet.

| Component | Status | In repo? | Verified | Notes |
|---|---|---|---|---|
| `kintsugi-cma` | Implemented | yes | 2,052 passed / 6 skipped (reported) | Docker build context fixed to `.` (was `./engine`, which does not exist). |
| `h-mem-temporal` | Implemented | yes | **26 passed** (re-run) | Time-aware retrieval with Ebbinghaus decay. |
| `tgs-verification` | Implemented | yes | **10 passed** (re-run) | Bidirectional text↔graph verification. |
| `tgs-rag-bridge` | Implemented | yes | imports OK | Requires `aiohttp` — now declared in `tgs-rag-bridge/requirements.txt`. |
| `sira-enrichment` | Implemented | yes | **17 passed** (re-run, was 14 + 3 failing) | Fixture bug fixed: content now uses the `example_user` trigger key. |
| `dispatch-notion-memory` | Implemented | yes | **11 passed** with `PYTHONPATH=src` (re-run) | `src/` layout, no packaging file — run/test with `PYTHONPATH=src`. Needs Notion creds at runtime. |
| `metacognition` | Implemented (in-tree) | yes | imports after installing jlens | Needs `jlens` from GitHub (not PyPI) **and** your own model + fitted J-lens. Default distilled model/lens pair is Coalition-internal. |
| `swarm` | Implemented | yes | runs | Requires external NATS + Ollama. Discord is now **opt-in** (`SWARM_DISCORD_ENABLED`); see `swarm/.env.example`. No hardcoded token path/channel. |
| `oracle-memory` | **External / private** | **no** (0 tracked files) | — | Referenced by README; not published in this public repo. |
| `kv-knowledge-packs` | **External / private** | **no** (0 tracked files) | — | Referenced by README; not published in this public repo. |
| `mnemosyne-metacognition` (standalone repo) | **External / private** | n/a | — | The GitHub URL 404s (private). The in-tree `metacognition/` is the reference implementation. |
| `hipporag-catrag-kg` | Design-only | spec only | — | `DESIGN.md` + `README.md`; no installable service. Other components that "assume" a KG service must supply their own. |
| `mnemosyne-wiki` | Design-only | spec only | — | `README.md` only. |
| `routines` | Design-only | spec only | — | `README.md` only. |

## External artifacts (not in this repo)

| Artifact | Where | Access |
|---|---|---|
| `jlens` / Jacobian Lens | https://github.com/anthropics/jacobian-lens | **Public**, but **not on PyPI**. Install: `pip install "git+https://github.com/anthropics/jacobian-lens.git"` |
| `Jackrong/Qwen3.5-27B-Claude-4.6-Opus-Reasoning-Distilled` + `opus_distill_jlens.pt` | referenced by `metacognition/test_metacognitive.py` defaults | **Private / Coalition-internal.** Supply your own model + lens via `--model` / `--lens`. |
| Public Qwen lens | Neuronpedia | Public, but fitted on **base `Qwen/Qwen3.5-27B`** — not a drop-in for the Opus-distilled default. |
| `oracle-memory`, `kv-knowledge-packs`, `mnemosyne-metacognition` | Liberation Labs (private) | Not in this public repo. |

## How to verify what IS here

```bash
# Core memory scaffold (idempotent):
bash setup.sh myagent            # auto-detects Claude Code / OpenClaw / Hermes / none

# Component test suites that pass from a clean checkout:
( cd kintsugi-cma          && python -m pytest -q )        # 2,052 passed / 6 skipped
( cd h-mem-temporal        && python -m pytest -q )        # 26 passed
( cd tgs-verification      && python -m pytest -q )        # 10 passed
( cd sira-enrichment       && python -m pytest -q )        # 17 passed
( cd dispatch-notion-memory && PYTHONPATH=src python -m pytest -q )  # 11 passed
```

## Known gaps (tracked in issue #4)

- No single full-stack installer yet: `setup.sh` provisions the **core memory
  scaffold** and prints an ordered per-component checklist; it does not stand up
  every service (several are external or design-only, above).
- `hipporag-catrag-kg`, `mnemosyne-wiki`, and `routines` are specifications.
  Components whose docs imply a live KG/wiki service must be pointed at an
  implementation you provide.
- OpenClaw integration is experimental and unverified — see the caveat in
  `AGENT_SETUP.md` (only Claude Code is currently supported end-to-end).
