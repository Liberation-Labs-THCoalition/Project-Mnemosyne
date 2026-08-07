# Precious Integration Audit — Fixes (Gus report)

Date: 2026-08-06/07. Fixes applied against HEAD `26805c4` (watermark reverted,
reproducibility fixes in). All five real bugs addressed; two additional
instances of the same bug families found and fixed along the way.

Test state after fixes:
- `kintsugi-cma`: **2,068 passed / 6 skipped** (was 2,052 / 6; +16 new tests)
- `h-mem-temporal`: **29 passed** (was 26; +3 new tests)

> **Process note (important):** while this work was in progress, a concurrent
> Claude session (`session_01M1JrjUigQoZyEfZuwU7MWT`) committed and pushed
> `3dd9d36` ("Fix real bugs from Precious integration audit"), sweeping up the
> then-current working tree: the fixes for #3/#6/#12-wiring/#2-dedup plus its
> own `MERGE_STRATEGY.md`. That commit's message undercounts what it contains
> (it says #3 and #6 only, and "1468 tests" — the files in it are the full fix
> set below minus the RLS migration and the dedup test). It is already on
> `origin/main` and `fork/main`, so it was left alone. The **remaining** work
> (RLS migration 002, agent-dedup tests, h-mem fix, COMPONENT_MANIFEST.md,
> this file) is staged and uncommitted per instructions.

---

## 1. Significance ordering reversed (finding #3) — FIXED

Convention (per docs, `org_isolation.py`, and the API): significance runs
1 = low/ephemeral to 10 = high/core.

**a) Hybrid retrieval symbolic query** — `kintsugi-cma/kintsugi/api/routes/memory.py`

The symbolic arm of `/api/memory/search` scored memories
`(10 - significance)/10` and ordered `significance ASC` — rewarding the
*least* significant memories. Now:

- score = `CAST(mu.significance AS FLOAT) / 10.0`
- `ORDER BY mu.significance DESC, mu.created_at DESC`
- SQL lifted to module-level `SYMBOLIC_SQL` (and made dialect-portable,
  `CAST` instead of `::float`) so tests can execute the exact shipped query.

**b) Same polarity bug in the expiration layers** —
`kintsugi-cma/kintsugi/memory/significance.py` (found during the fix)

`compute_layer()` mapped significance 1-2 → PERMANENT (never expires) and
9-10 → VOLATILE (reaped after 30 days): the reaper would have archived an
org's *most important* memories after a month while keeping throwaway notes
forever. Mapping flipped to match the documented convention
(1-2 → VOLATILE/30d ... 9-10 → PERMANENT/never); contradicted
`org_isolation.py` in the same package, which already used `>= 8 → core`.

**Tests:** new `kintsugi-cma/tests/test_api_memory_ranking.py` (5 tests) runs
`SYMBOLIC_SQL` against a real database and proves high-significance ranks
above low, scores are monotone in significance, ties break by recency, org
scoping holds, and `LIMIT` keeps the most significant rows.
`test_memory_significance.py` / `test_phase1_integration.py` updated for the
corrected layer mapping.

## 2. RLS not on live tables (finding #12) — FIXED (Postgres) + DOCUMENTED (SQLite)

RLS existed only in the standalone `org_memories` module
(`kintsugi/memory/org_isolation.py`), which the live API never uses. The API
operates on `memory_units` and siblings — which had no RLS at all.

- **New migration** `kintsugi-cma/migrations/versions/002_rls_live_tables.py`:
  ENABLE + FORCE row-level security and per-operation policies
  (`org_id::text = current_setting('app.current_org_id', true)`) on
  `memory_units`, `temporal_memories`, `memory_archives`, `intent_capsules`,
  `shield_constraints`; child tables `memory_embeddings` / `memory_lexical` /
  `memory_metadata` scoped through their parent `memory_units` row via
  EXISTS. Mirrors the `org_memories` pattern, including FORCE (the app
  connects as the table owner, so without FORCE the policies would be
  decorative). `organizations` deliberately stays open — it is the registry
  used to validate org ids before binding context.
- **Context wiring**: new `kintsugi.db.set_org_context(session, org_id)`
  (transaction-local `set_config`, no-op on non-Postgres). Called after org
  validation in `/api/memory/search`, `/api/memory/store`,
  `/api/agent/message`, `/api/agent/temporal`. Fail-closed: a transaction
  that skips it sees empty tables and cannot write.
- **Honest documentation** in `COMPONENT_MANIFEST.md`: the SQLite seed tier
  has no RLS mechanism at all — isolation there rests on application-level
  `org_id` WHERE clauses only. Also documented: background jobs/scripts must
  call `set_org_context` once migration 002 is applied.

Not runtime-verified against a live Postgres (none in this environment);
migration compiles, chain resolves `001 → 002 (head)`, and statements follow
the proven `org_memories` pattern. Flagging for a staging `alembic upgrade`
before production.

## 3. UK/international phone detection (finding #6) — FIXED

`kintsugi-cma/kintsugi/security/pii.py` had one US/NANP phone regex;
"+44 7700 900123" was invisible in mixed content.

- Added a second PHONE pattern for E.164/international numbers:
  `(?<![\d+])\+\d{1,3}(?:[-.\s]?\d{2,4}){2,5}(?!\d)` — catches
  `+44 7700 900123`, `+447700900123`, `+44-20-7946-0958`, `+14155552671`,
  including *grouped* forms the audit's literal suggestion
  (`\+\d{1,3}\s?\d{4,14}`) would only half-redact (it stops at the second
  space, leaving `900123` in the "redacted" text).
- `detect()` now resolves same-type overlapping matches (a `+1 ...` number
  hits both phone patterns) to the single longest finding, so redaction never
  emits doubled `[REDACTED_PHONE][REDACTED_PHONE]` tokens. Cross-type
  overlaps still report both findings (existing contract, covered by
  `test_redact_remove_adjacent_pii`).

**Tests:** new `TestPIIPhoneInternational` class (7 tests) in
`tests/test_security_pii.py`, including the audit's mixed-content case (UK
number alongside email and US number) and a no-digits-survive redaction
check.

## 4. H-MEM dreamer_consolidator (finding #7) — CONFIRMED OURS TOO, FIXED

Our `h-mem-temporal/dreamer_consolidator.py` had the same bug Gus fixed
locally: `_generate_summary()` and `_check_relationship()` called
`llm_generate(prompt, ollama_url=self.ollama_url)` but `llm_generate()` only
accepted `(prompt, system, timeout)` — a guaranteed `TypeError` at runtime
(masked in tests because they patch `llm_generate`), and the instance `model`
setting was silently ignored in favor of the env default.

- `llm_generate()` now accepts optional `ollama_url` / `model`, defaulting to
  the module-level env config.
- Both call sites pass `ollama_url=self.ollama_url, model=self.model`.

**Tests:** 3 new tests mock `requests.post` (not `llm_generate`) and verify
the instance URL and model actually reach the HTTP call, plus env-default
behavior.

## 5. Duplicate temporal events (finding #2 note) — FIXED

`POST /api/agent/message` inserted a `temporal_memories` row unconditionally,
so client retries/double-submits duplicated events.

- `kintsugi-cma/kintsugi/api/routes/agent.py`: before inserting, the handler
  looks for an identical `(org, category, message)` event within
  `TEMPORAL_DEDUP_WINDOW_SECONDS` (60s) and reuses it — on both the blocked
  and allow/warn paths. Response gains an additive `deduplicated: bool`
  field; `temporal_event_id` points at the original event.

**Tests:** new `kintsugi-cma/tests/test_api_agent_dedup.py` (4 tests) runs
the real route over in-memory SQLite: duplicate reuses the event (1 row),
distinct messages don't, duplicates outside the window create a new event,
and dedup is org-scoped.

---

## Enabling changes (needed to test the fixes at all)

- `kintsugi-cma/kintsugi/db.py`: engine/sessionmaker now created **lazily**
  instead of at import time. Previously `import kintsugi.models` (or any
  route) crashed without a DB driver installed — which is why zero
  route/model tests existed. `engine` / `async_session` stay importable via
  module `__getattr__`; `main.py` lifespan behavior unchanged.
- `aiosqlite` added to `kintsugi-cma/requirements-test.txt` (and installed in
  the venv) for the SQLite-backed route tests. `pgvector` was also missing
  from the venv (it is in `pyproject.toml`) and was installed.

## Additional finding (documented, not fixed)

The SQLite seed tier (`docker-compose.seed.yml`) cannot provision its schema:
migration 001 uses Postgres-only constructs (`CREATE EXTENSION vector`,
JSONB/TSVECTOR/pgvector types) and the ORM's JSONB columns have no SQLite DDL
rendering (`CompileError` on `create_all`). Recorded under Known gaps in
`COMPONENT_MANIFEST.md`; needs a dialect-aware schema story if the seed tier
is a real deployment target.

## Review state

- Committed & pushed by the concurrent session (`3dd9d36`): routes/memory.py,
  routes/agent.py, db.py, significance.py, pii.py, requirements-test.txt,
  test_api_memory_ranking.py, test_memory_significance.py,
  test_phase1_integration.py, test_security_pii.py, MERGE_STRATEGY.md.
- **Staged, uncommitted (this session, for review):**
  `kintsugi-cma/migrations/versions/002_rls_live_tables.py`,
  `kintsugi-cma/tests/test_api_agent_dedup.py`,
  `h-mem-temporal/dreamer_consolidator.py`, `h-mem-temporal/test_hmem.py`,
  `COMPONENT_MANIFEST.md`, `GUS_AUDIT_FIXES.md`.
- Note: because of the split above, upstream currently has the RLS *wiring*
  and the dedup *feature* without the RLS *migration* and dedup *tests*
  (harmless — `set_org_context` is a no-op until migration 002 is applied —
  but the staged remainder should land promptly).
