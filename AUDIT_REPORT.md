# Functional Completeness Audit — 2026-08-04

Follow-up verification for issue #4 (Gus: "some components are just design docs").
Every component in `COMPONENT_MANIFEST.md` was re-verified from the working tree
at commit `a878b49`. Method: `git ls-files` per component (tracked files only),
`py_compile` sweep over all tracked Python, test suites re-run, AST-based
third-party import scan, invisible-unicode scan, secrets/path grep, and a
sandboxed end-to-end run of `setup.sh` (with `HOME` pointed at a scratch dir).

## Verdict: manifest is honest. All "Implemented" rows run; all "Design-only" rows are specs.

| Component | Runnable code | Compiles | Tests | Deps declared | Hardcoded config |
|---|---|---|---|---|---|
| kintsugi-cma | yes (196 .py) | yes | **2,052 passed / 6 skipped** (71s) | yes — `requirements-test.txt` suffices for the suite; runtime deps in `pyproject.toml` | none (dev-compose Postgres creds `kintsugi/kintsugi` are local-compose defaults) |
| h-mem-temporal | yes (4 .py) | yes | **26 passed** | was missing `requests` — **fixed** (new `requirements.txt`) | none (`OLLAMA_URL` env-overridable) |
| tgs-verification | yes (5 .py) | yes | **10 passed** | stdlib-only | none |
| tgs-rag-bridge | yes (1 .py) | yes | none exist — imports OK | yes (`aiohttp` in `requirements.txt`) | README mentions MTH deployment (descriptive only) |
| sira-enrichment | yes (2 .py) | yes | **17 passed** | was missing `requests` — **fixed** (new `requirements.txt`) | none |
| dispatch-notion-memory | yes (16 .py) | yes | **11 passed** (`PYTHONPATH=src`) | yes (`requirements.txt`; starlette/uvicorn arrive transitively via `mcp`) | example config shipped real Notion DB UUIDs — **fixed** (placeholders) |
| metacognition | yes (6 .py) | yes | can't run without model/lens (documented) | yes — incl. `jlens @ git+…` (URL verified live, HTTP 200) | defaults name Coalition-internal model, overridable via `--model`/`--lens` (documented in manifest) |
| swarm | yes (3 .py) | yes | none exist — all 3 modules import | was missing `requests`, `nats-py` — **fixed** (new `requirements.txt`) | `NATS_USER=nexus` default only, env-overridable and documented in `.env.example`; Discord opt-in confirmed in code |
| hipporag-catrag-kg | **no — design-only** (0 .py; `DESIGN.md` + `README.md`) | n/a | n/a | n/a | n/a — manifest status correct |
| mnemosyne-wiki / routines | no — README-only specs | n/a | n/a | n/a | manifest status correct |
| oracle-memory / kv-knowledge-packs | 0 tracked files (local dirs hold only ignored venv/pytest residue; absent from a fresh clone) | n/a | n/a | n/a | manifest "External / private" correct |

Test totals independently reproduced on this machine (Python 3.12.3) and match
the manifest's claimed numbers exactly. kintsugi-cma's count was previously
"(reported)"; it is now verified: 2,058 collected, 2,052 passed, 6 skipped.
The suite passes without `pgvector`/`asyncpg`/`celery` installed — those are
runtime-only (Docker) deps and the tests never touch the DB models layer.

## Cross-cutting checks

- **Watermark residue: none.** All 233 tracked `.py` files pass `py_compile`.
  Zero-width/invisible-unicode scan over tracked `.py/.md/.sh/.yml/.toml` files
  found nothing; no `watermark`/`stego` strings remain.
- **Imports of removed modules: none.** No tracked Python imports
  `oracle-memory`, `kv-knowledge-packs`, or anything from them. Three *doc*
  pointers to files a clone doesn't have were found and fixed (below).
- **`setup.sh`: works end-to-end.** Verified in a sandboxed `$HOME` in both
  `none` and `claude` scaffold modes; creates only the advertised directories,
  no network access, idempotent re-run exits 0. (`--venv` opt-in path not
  exercised to avoid creating `.venv` in the working tree.)
- **Secrets: none.** All token-like strings are docstring examples or test
  fakes (`xoxb-fake-token-for-testing-only` etc.).
- **CI** runs kintsugi-cma only (3.11 + 3.12 matrix); other suites are manual.

## Changes staged for review (NOT committed)

1. `h-mem-temporal/requirements.txt` — new; declares `requests` (+ pytest).
2. `sira-enrichment/requirements.txt` — new; declares `requests` (+ pytest).
3. `swarm/requirements.txt` — new; declares `requests`, `nats-py`.
4. `dispatch-notion-memory/config/config.example.yaml` — replaced four real
   Notion database UUIDs with empty placeholders (user-specific config was
   baked into the tracked example).
5. `AGENT_SETUP.md` — three stale out-of-repo pointers fixed: the dreamer
   "Memory System Spec" link (`../nexus-memory-archive/…`), the two-graph
   `MEMORY_ARCHITECTURE.md` link (`../agents/nexus/…`), and the Layer 6
   instruction to read `kv-knowledge-packs/kv_packs.py` (external artifact) —
   each now states the artifact is not in this repo and points at what is.
6. `COMPONENT_MANIFEST.md` — kintsugi count upgraded from "(reported)" to
   verified; per-row notes for the new requirements files and "no test suite"
   flags (tgs-rag-bridge, swarm); audit date added; three new Known-gaps
   entries (CI scope, `.gitignore`'s `dispatch-notion-memory/` footgun,
   kintsugi's diverged `requirements.txt` vs `pyproject.toml`).
7. `AUDIT_REPORT.md` — this file.

No status upgrades or downgrades were warranted: nothing marked Implemented
failed, and no Design-only component contained hidden code.

## Recommendations (not acted on)

- Decide on one dependency source of truth for kintsugi-cma, or add a comment
  cross-referencing the two; consider `[project.optional-dependencies]` extras
  for the lazily imported Slack adapter deps (`slack_sdk`, `aiohttp`).
- Remove the blanket `dispatch-notion-memory/` line from `.gitignore` (new
  files there are silently ignored) in favor of specific entries.
- Add the small suites (h-mem, tgs-verification, sira, dispatch) to CI — they
  finish in ~4s combined.
- `datetime.utcnow()` deprecation warnings appear in kintsugi-cma and
  dispatch-notion-memory tests; will break on a future Python.
- Add smoke tests for `tgs-rag-bridge` and `swarm` (currently import-verified
  only).
