# Component Manifest

Honest status of every component advertised by Project Mnemosyne, so a fresh
clone knows what is runnable now, what is external/gated, and what is still a
design spec. Created in response to issue #4 (reproducibility).

Last full audit: **2026-08-04** — every table row below was re-verified from the
working tree on that date (see `AUDIT_REPORT.md` for method and details).

**Legend**
- **Implemented** — source is in this repository and runs/tests from a clean checkout.
- **External / private** — referenced by the docs but **not present in this public repo** (gated, private, or a separate artifact).
- **Design-only** — specification / README in-tree; no runnable service yet.

| Component | Status | In repo? | Verified | Notes |
|---|---|---|---|---|
| `kintsugi-cma` | Implemented | yes | **2,068 passed / 6 skipped** (verified 2026-08-06, 62s) | Docker build context fixed to `.` (was `./engine`, which does not exist). Test suite needs only `requirements-test.txt`; full runtime deps live in `pyproject.toml` (used by the Docker build). Precious audit fixes applied 2026-08-06: significance ranking polarity, RLS on live tables (see below), E.164 phone PII, temporal-event dedup. |
| `h-mem-temporal` | Implemented | yes | **29 passed** (verified 2026-08-06) | Time-aware retrieval with Ebbinghaus decay. `requests` (used by `dreamer_consolidator.py`) now declared in `h-mem-temporal/requirements.txt`. Audit fix: `DreamerConsolidator` instance `ollama_url`/`model` are now actually passed to `llm_generate` (previously a TypeError at runtime). |
| `tgs-verification` | Implemented | yes | **10 passed** (verified 2026-08-04) | Bidirectional text↔graph verification. Stdlib-only — no third-party runtime deps. |
| `tgs-rag-bridge` | Implemented | yes | imports OK (verified 2026-08-04) | Requires `aiohttp` — declared in `tgs-rag-bridge/requirements.txt`. No test suite yet. |
| `sira-enrichment` | Implemented | yes | **17 passed** (verified 2026-08-04) | Fixture bug fixed: content now uses the `example_user` trigger key. `requests` now declared in `sira-enrichment/requirements.txt`. |
| `dispatch-notion-memory` | Implemented | yes | **11 passed** with `PYTHONPATH=src` (verified 2026-08-04) | `src/` layout, no packaging file — run/test with `PYTHONPATH=src`. Needs Notion creds at runtime. `config/config.example.yaml` ships placeholder database IDs — fill in your own. |
| `metacognition` | Implemented (in-tree) | yes | compiles; imports need `jlens` (declared in requirements.txt via git URL) | Needs `jlens` from GitHub (not PyPI) **and** your own model + fitted J-lens (`--model` / `--lens`). Default distilled model/lens pair is Coalition-internal. `cognitive_snapshot` imports standalone. |
| `swarm` | Implemented | yes | imports OK (verified 2026-08-04) | Requires external NATS + Ollama. Discord is now **opt-in** (`SWARM_DISCORD_ENABLED`); see `swarm/.env.example`. No hardcoded token path/channel. Pip deps (`requests`, `nats-py`) declared in `swarm/requirements.txt`. No test suite yet. |
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
( cd kintsugi-cma          && python -m pytest -q )        # 2,068 passed / 6 skipped
( cd h-mem-temporal        && python -m pytest -q )        # 29 passed
( cd tgs-verification      && python -m pytest -q )        # 10 passed
( cd sira-enrichment       && python -m pytest -q )        # 17 passed
( cd dispatch-notion-memory && PYTHONPATH=src python -m pytest -q )  # 11 passed
```

## Row-level security in kintsugi-cma (audit finding #12)

Honest state after the 2026-08-06 fixes:

- **PostgreSQL (sprout/grove tiers):** RLS is now on the tables the live API
  actually uses. Migration `002_rls_live_tables.py` enables **ENABLE + FORCE**
  row-level security with per-operation policies on `memory_units`,
  `temporal_memories`, `memory_archives`, `intent_capsules`, and
  `shield_constraints` (keyed on `org_id`), and on the child tables
  `memory_embeddings` / `memory_lexical` / `memory_metadata` (scoped through
  their parent `memory_units` row). This mirrors the pattern that previously
  existed only in the standalone `org_memories` module
  (`kintsugi/memory/org_isolation.py`), which the API never used.
- **Operational contract:** because RLS is FORCEd, *every* transaction that
  touches these tables must first bind an org via
  `kintsugi.db.set_org_context(session, org_id)` (sets the transaction-local
  `app.current_org_id`). The API routes do this after validating the org.
  Background jobs or scripts that skip it will see empty tables and be unable
  to write — that is the fail-closed design, not a bug. `organizations` is
  deliberately outside RLS (it is the registry consulted to validate org ids).
- **SQLite (seed tier):** SQLite has no row-level security. Isolation on the
  seed tier rests solely on application-level `org_id` WHERE clauses;
  migration 002 and `set_org_context` are explicit no-ops there.

## Known gaps (tracked in issue #4)

- No single full-stack installer yet: `setup.sh` provisions the **core memory
  scaffold** and prints an ordered per-component checklist; it does not stand up
  every service (several are external or design-only, above).
- `hipporag-catrag-kg`, `mnemosyne-wiki`, and `routines` are specifications.
  Components whose docs imply a live KG/wiki service must be pointed at an
  implementation you provide.
- OpenClaw integration is experimental and unverified — see the caveat in
  `AGENT_SETUP.md` (only Claude Code is currently supported end-to-end).
- CI (`.github/workflows/ci.yml`) runs the `kintsugi-cma` suite only; the other
  component suites are verified manually (commands above).
- `.gitignore` contains a blanket `dispatch-notion-memory/` entry: the 21
  already-tracked files stay tracked, but **new** files added under that
  directory are silently ignored unless force-added.
- `kintsugi-cma/requirements.txt` (test/dev path) and `pyproject.toml`
  (runtime/Docker path) have diverged; each covers its own path, but they are
  two sources of truth. Optional adapter deps (`slack_sdk`, `aiohttp` for Slack
  OAuth) are lazily imported and not declared as extras.
- The seed tier (`docker-compose.seed.yml`, SQLite) cannot provision its
  schema from the current migration chain: `001_initial` uses
  Postgres-only constructs (`CREATE EXTENSION vector`, JSONB/TSVECTOR/pgvector
  column types), and the ORM's JSONB columns have no SQLite DDL rendering.
  Found while adding SQLite-backed route tests for the audit fixes; the SQLite
  path works for the SQL the tests create by hand, but a real seed deploy
  needs a dialect-aware schema story.
