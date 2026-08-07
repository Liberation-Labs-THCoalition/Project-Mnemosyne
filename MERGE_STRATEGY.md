# Merge Strategy: origin/main <-> fork/main Sync

**Repo:** Project-Mnemosyne (local checkout: `~/Agent-Memory-Architectures/`)
**origin:** `Liberation-Labs-THCoalition/Project-Mnemosyne` (public, canonical) — tip `9c3f701`
**fork:** `HumboldtJoker/Project-Mnemosyne` (public, standalone repo — NOT a GitHub-network fork; `"fork": false` per API) — tip `0490614`
**Status:** PLANNING DOCUMENT. Nothing here has been executed.
**Prepared:** 2026-08-03 by Nexus. All numbers below verified against the local checkout with both remotes fetched.

---

## 1. Executive Summary

The briefing assumed the fork carried months of unmerged work. **It does not.** The measured topology inverts the picture:

- **Merge-base:** `1707bcbe3b959ca1f289cfc50ce0841b7a55d4fc` ("Fix 7 metadata drift vectors in Dispatch memory pipeline", 2026-06-12).
- **fork/main is exactly 1 commit ahead** of the merge-base: `0490614` (its copy of the security redaction).
- **origin/main is 14 commits ahead**: security fix, private-repo curation, metacognition restore, VERSION baseline, licensing, watermark apply+revert (net no-op), reproducibility fixes.
- The Dispatch/Notion pipeline, Scout training lab notebook, Dreamer pruner (`h-mem-temporal/PRUNER_SPEC.md`, `dreamer_consolidator.py`), TGS, kintsugi, swarm — **all shared history below the merge-base.** Present in both repos already.
- The "163 files / 31K lines" divergence is **origin's cleanup, not fork's additions**: origin deleted 138 files (~31K lines of dual-use experimental code moved to the private `jlens-experiments` repo in `6a118fa`) that the fork still exposes at HEAD.

**Consequence:** this is not a hard merge. A standard 3-way merge produces **exactly one conflict** (verified with `git merge-tree`), does **not** resurrect anything origin deleted, and — once that conflict is resolved as "delete" — yields a tree **byte-identical to origin/main today**. The sync is pure history unification for origin and a pure content upgrade (plus security remediation) for the fork.

---

## 2. Merge-Base and Branch State

| Item | Value |
|---|---|
| Merge-base | `1707bcbe3b959ca1f289cfc50ce0841b7a55d4fc` |
| origin/main tip | `9c3f701` "Fix reproducibility blockers and revert broken watermark (issues #3, #4)" |
| fork/main tip | `0490614` "fix(security): remove Tailscale IPs, SSH tunnel details, and NATS default password" |
| Commits ahead (origin) | 14 |
| Commits ahead (fork) | 1 |
| Local `main` | `9c3f701`, in sync with origin/main, worktree clean |
| Other branches | `dispatch-notion-memory` exists on both remotes at identical tip `b9ad023` — already synced, out of scope. `origin/nexus/tgs-sira-v2` origin-only, out of scope. |

origin/main commits since merge-base (oldest first):

```
79a4abe  Exclude dispatch-notion-memory from kintsugi-cma CI
e51c287  Fix 2 async test failures on Python 3.12
08b9c78  Add ablation study publication draft
00e5333  Add Metacognitive Memory layer to Mnemosyne stack
cecdf4f  Add metacognitive memory modules (J-lens, circumplex, ghost tracking)
9573f26  fix(security): remove Tailscale IPs, SSH tunnel, NATS default password
6a118fa  Move sensitive experimental code to private repo  <- the 31K-line deletion
a82cd87  Restore metacognitive measurement probes to public repo
1c6f7df  Add VERSION.md — v0.1.0 baseline
9a6852a  Add Hermes scaffold support to AGENT_SETUP.md
2f97847  Require express written approval for sliding scale
fab6d32  Apply steganographic watermarks (227 files, +-2507)
bfeffb7  Revert watermarks
9c3f701  Fix reproducibility blockers (setup.sh, .env.example, COMPONENT_MANIFEST)
```

fork/main commits since merge-base: `0490614` only.

---

## 3. Divergence Analysis

### 3.1 merge-base -> origin/main: 164 files, +2,365 / -31,220

By status: **138 D, 14 A, 12 M.** By top-level directory:

| Dir | D | A | M | Notes |
|---|---|---|---|---|
| kv-knowledge-packs/ | 122 | — | — | Entire directory removed (31 ethics-pack dirs = 91 files, 10 experiment_results JSONs, 21 root scripts/docs) |
| oracle-memory/ | 16 | — | — | Entire directory removed |
| metacognition/ | — | 8 | — | New: probes restored after review (`a82cd87`) |
| (root) | — | 3 | 5 | + VERSION.md, COMPONENT_MANIFEST.md, setup.sh; M: .gitignore, AGENT_SETUP.md, LICENSE-COMMERCIAL-ADDENDUM.md, README.md, TRAINING_LAB_NOTEBOOK.md |
| kintsugi-cma/ | — | — | 4 | docker-compose x2, tests x2 |
| swarm/ | — | 1 | 1 | + .env.example; M swarm_service.py (security + repro) |
| publications/ | — | 1 | — | ablation-study-draft.md |
| tgs-rag-bridge/ | — | 1 | — | requirements.txt |
| sira-enrichment/ | — | — | 1 | test_sira.py |
| dispatch-notion-memory/ | — | — | 1 | README.md |

### 3.2 merge-base -> fork/main: 2 files, +5 / -5

| File | Change |
|---|---|
| `TRAINING_LAB_NOTEBOOK.md` | Redacted Tailscale/transfer details (3 hunk-lines) |
| `kv-knowledge-packs/EXPERIMENT_METHODS.md` | Redacted Studio Tailscale endpoints to `[internal-tailscale]` (2 hunk-lines) |

### 3.3 Tip-to-tip (origin/main -> fork/main): 163 files, +31,217 / -2,362

138 A (fork still has what origin deleted) + 14 D (fork lacks origin's new files) + 11 M (fork lacks origin's edits). This matches the briefing's "163 files, 31K+ lines" — but the delta is almost entirely origin's curation, mirrored.

---

## 4. Conflict Candidates

Files modified on BOTH sides since merge-base — **exactly two**, and only one actually conflicts:

| File | origin side | fork side | Merge outcome |
|---|---|---|---|
| `TRAINING_LAB_NOTEBOOK.md` | Redacted in `9573f26` | Redacted in `0490614` | **Auto-resolves.** Blobs identical at both tips (`5da1d39`) — both sides made the same change. |
| `kv-knowledge-packs/EXPERIMENT_METHODS.md` | Redacted in `9573f26`, then **deleted** in `6a118fa` (moved to private repo) | Redacted in `0490614`, still present | **CONFLICT (modify/delete)** — the only one. Verified via `git merge-tree --write-tree origin/main fork/main` (same single conflict in both merge directions; auto-merged tree `1526873e`). |

**Resolution is lossless:** fork's redacted blob and the blob origin deleted are byte-identical (`36d3d26` on both sides — the two machines applied the same redaction two seconds apart on 2026-07-15). Deleting the file discards nothing that isn't already preserved at origin commit `9573f26` and in `Liberation-Labs-THCoalition/jlens-experiments` (private). **Resolve as DELETE** to uphold the `6a118fa` curation decision.

**Verified:** with the conflict resolved as delete, the merged tree is **byte-identical to origin/main's current tree** (`git diff origin/main 1526873e` shows only that one file).

### 4.1 Security-fix overlap (briefing concern, resolved)

`0490614` (fork) and `9573f26` (origin) share a commit message but have different patch-ids. Origin's version covers **one additional file**: `swarm/swarm_service.py`, removing the NATS default password. The fork's fix missed it:

```
fork/main:swarm/swarm_service.py:33:   NATS_PASS = os.environ.get("NATS_PASS", "op-nats-changeme")   <- still live on public fork
origin/main:swarm/swarm_service.py:33: NATS_PASS = os.environ.get("NATS_PASS", "")
```

Fork didn't touch this file after the merge-base, so the merge takes origin's version cleanly — **the merge itself is the remediation.** (NATS credentials were rotated 2026-07-17; confirm before treating this as low urgency — see Risk R1.)

### 4.2 Watermark revert (briefing concern, resolved)

`fab6d32` (apply, 227 files) + `bfeffb7` (revert) is a **perfect net no-op**: `git diff fab6d32^ bfeffb7` is empty. It contributes nothing to any 3-way merge; it enters the fork's history as inert commits. Nothing "needs to apply cleanly" — it already cancelled on origin's side.

---

## 5. Categorization of Fork-Only Content (138 files)

All 138 fork-only files are content origin **deliberately removed** in `6a118fa`: "dual-use research tools for model steering and internal state measurement," moved to private `jlens-experiments`. The governing precedent is `a82cd87`: read-only *measurement* tools may return to public after review; *injection/steering* code stays private. Categories below follow that policy.

### (a) Should go to origin

- **`0490614` itself (the commit, not new content).** Its tree delta relative to origin is zero after conflict resolution, but merging it records Thomas's parallel security fix in origin's history and unifies the DAG. This is the only fork-side item origin lacks.
- **Restoration candidates — only via the `a82cd87` review path, NOT via this merge:** `graph_encoder.py`, `graph_benchmark.py`, `graph_benchmark_powered.py`, `graph_experiment.py`, `graph_experiment_phase2.py`, `test_kvpack.py`, `judge_scorer.py`, `llm_rescorer.py` — reproduction/evaluation tooling for the published Graph Topology paper (repro value; scanned clean of endpoints). If restored, restore from `jlens-experiments` in a dedicated reviewed commit, as was done for `metacognition/`.

### (b) Should stay out of the public org repo (and, recommendation: off the public fork's HEAD too)

- **`kv-knowledge-packs/ethics_packs/`** — 31 pack dirs, 91 files. Bulk data derived from the Stanford Encyclopedia of Philosophy (copyrighted; redistribution of derived triples/walk encodings is a licensing exposure) plus repo bloat.
- **`kv-knowledge-packs/experiment_results/`** — 10 raw run JSONs. Lab artifacts, not library code.
- **Injection/steering code (explicitly named private by `6a118fa`):** `kv_packs.py`, `direct_tensor_injection.py`, `value_only_injection.py`, `blend_ablation.py`, `powered_decomp_study.py`, `ethics_pack_builder.py`, `muse_values.py` (Muse ethical-KV-injection framework — Coalition-internal), `geometry_observer.py`.
- **`oracle-memory/` (16 files)** — internals of the unreleased Oracle model, including `NEXUS_REVISION_SPEC.md` which hardcodes `/home/admin/...` paths (lines 254, 261).

### (c) Needs review before any future restoration

- **`kv-knowledge-packs/EXPERIMENT_METHODS.md`** — even post-redaction it is an internal ops document: `/mnt/data1/...` data paths, machine names (Margaret, Cairn, Studio, MTH), OAuth-refresh cron cadence, internal file map. Recommend it stays private (delete-resolution in Section 6).
- **`kv-knowledge-packs/FINDINGS.md`, `kv-knowledge-packs/README.md`** — scanned clean of IPs/paths, but they document injection methodology; review under the dual-use policy.
- **`studio_scale_test.py`, `tgs_kvpack_bridge.py`** — clean (env-var endpoints, localhost defaults), but they are drivers for the injection pipeline.

**Secret-scan results across full fork tip (verified):** no raw Tailscale CGNAT IPs (100.64-127.x), no live `*.ts.net` hostnames in file content, no real credentials (all `password=`/`token=` hits are docstring examples and test fixtures in kintsugi-cma). The three genuine residuals: `op-nats-changeme` default (Section 4.1), `/home/admin` paths in `oracle-memory/NEXUS_REVISION_SPEC.md`, and machine names in the (deliberately retained, already-public) `TRAINING_LAB_NOTEBOOK.md`. Note: `kintsugi-cma/coverage.xml` carries `/home/admin` paths on **both** tips — pre-existing, not merge-blocking, follow-up in R5.

---

## 6. Recommended Merge Approach

**Merge commit, not rebase. Direction: merge `fork/main` INTO `origin/main`, then fast-forward the fork to the same commit.**

Rationale:

1. **Merge, not rebase:** both histories are published on public repos. Rebasing origin's 14 commits is out of the question (rewrites release history, breaks issue refs #3/#4). Rebasing the fork's single commit onto origin degenerates to an *empty commit* (both its files already match origin or are deleted) — i.e., a force-push that erases Thomas's security-fix attribution from the fork's line. A merge commit needs no force-push anywhere and preserves attribution.
2. **Into origin:** origin is canonical (releases, VERSION baseline). Merging fork->origin puts origin's line as first-parent, keeping `git log --first-parent` release history clean. Content-wise the direction is symmetric — `git merge-tree` produces the identical tree and identical single conflict both ways.
3. **Fork syncs by fast-forward:** `fork/main` (`0490614`) is a parent of the merge commit, so pushing that commit to the fork is a clean fast-forward. Both repos converge on one commit, one tree — identical to origin's current tree. Divergence goes to zero without any force-push.
4. **Curation is preserved automatically:** because the fork made no post-base changes to any file origin deleted, the 3-way merge keeps all 138 deletions. Nothing sensitive resurfaces. Verified by simulation.

**Option B (rejected, documented for completeness):** if Thomas wants the experimental content to remain at his fork's HEAD, resolve the conflict keep-fork-side and follow the merge with `git checkout 0490614 -- kv-knowledge-packs oracle-memory` on the fork only. **Not recommended:** the fork is public, so this perpetuates public HEAD exposure of exactly the dual-use content `6a118fa` withdrew, plus the SEP-derived data licensing exposure. If Thomas wants this content close at hand, the private `jlens-experiments` repo already holds it; a private fork/branch of that is the right home. This decision is his to make — flag it before executing.

---

## 7. Files to Exclude from the Merge

No pathspec-level exclusion machinery is needed — the 3-way merge already excludes everything by keeping origin's deletions. Exclusion reduces to **one decision**:

| File | Action | Why |
|---|---|---|
| `kv-knowledge-packs/EXPERIMENT_METHODS.md` | `git rm` during conflict resolution | Internal ops doc; canonical home is `jlens-experiments`. Fork's redacted blob is byte-identical to `9573f26`'s — zero information loss. |

Sanity checks written into the procedure: the resolved tree must be byte-identical to pre-merge `origin/main` (step 9), and post-merge fork HEAD must not contain `op-nats-changeme`, `ethics_packs/`, or `oracle-memory/` (step 13).

---

## 8. Step-by-Step Procedure

Executor: anyone with push rights to both repos. Estimated time: 15 minutes. All steps from `~/Agent-Memory-Architectures/`.

**Phase 0 — Preflight**

1. `git fetch origin --no-tags && git fetch fork --no-tags`
2. Verify the world matches this spec (ABORT and re-analyze if any differ):
   ```
   git rev-parse origin/main          # expect 9c3f7016e186b757dd246d9c4ef6c4e433648900
   git rev-parse fork/main            # expect 049061420aef2467044ca34f9820c55e452b0547
   git merge-base origin/main fork/main   # expect 1707bcbe3b959ca1f289cfc50ce0841b7a55d4fc
   git status --porcelain             # expect empty
   ```
3. Confirm with Thomas: (a) fork content policy — Option A (recommended: fork HEAD converges to origin, experimental content lives in private jlens-experiments) vs Option B (Section 6); (b) NATS credentials were rotated after 2026-07-15 (session log says 2026-07-17 — verify, don't assume).
4. Confirm `jlens-experiments` (private) actually contains the moved content: at minimum `EXPERIMENT_METHODS.md`, `kv-knowledge-packs/` scripts, `oracle-memory/`. This is the loss-prevention backstop.
5. Safety tag on the fork's pre-sync state (cheap rollback + permanent reachability):
   ```
   git tag fork-pre-sync-2026-08 fork/main
   git push fork fork-pre-sync-2026-08
   ```

**Phase 1 — Merge on an integration branch**

6. ```
   git checkout -b sync/fork-origin-2026-08 origin/main
   git merge --no-ff fork/main
   ```
   Expect exactly: `CONFLICT (modify/delete): kv-knowledge-packs/EXPERIMENT_METHODS.md`. **Any other conflict means reality has drifted from this spec — abort (`git merge --abort`) and re-analyze.**
7. Resolve as delete:
   ```
   git rm kv-knowledge-packs/EXPERIMENT_METHODS.md
   ```
8. Commit with an explanatory message:
   ```
   git commit -m "Merge fork/main: unify histories after parallel security fixes

   Both repos applied the same Tailscale/credential redaction on 2026-07-15
   (origin 9573f26, fork 0490614). Origin's fix additionally cleared the NATS
   default password in swarm/swarm_service.py; this merge brings that plus all
   post-June work (metacognition, VERSION baseline, reproducibility fixes,
   private-repo curation per 6a118fa) to the fork.

   Conflict resolution: kv-knowledge-packs/EXPERIMENT_METHODS.md deleted,
   matching 6a118fa (canonical copy lives in jlens-experiments, private).
   Fork's redacted blob was byte-identical to 9573f26's — nothing lost."
   ```
9. **Tree invariant check (must pass):**
   ```
   git diff --quiet origin/main HEAD && echo "TREE IDENTICAL — OK" || echo "STOP: tree differs from origin/main"
   ```
10. Optional but cheap: `bash setup.sh` / run the kintsugi + sira test suites. Low value here (tree is unchanged from origin, which is presumed green), but it validates the checkout.

**Phase 2 — Publish**

11. Push to origin: `git push origin sync/fork-origin-2026-08:main`
    (If org branch protection requires a PR: push the branch, open PR "Sync fork histories", merge with a **merge commit** — not squash, not rebase — to preserve the dual-parent DAG.)
12. Push the same commit to the fork — this is a fast-forward, **no force flag**:
    ```
    git push fork sync/fork-origin-2026-08:main
    ```
    If the push is rejected as non-fast-forward, someone moved fork/main since preflight — STOP, refetch, re-run Phase 0.

**Phase 3 — Verify and clean up**

13. ```
    git fetch origin --no-tags && git fetch fork --no-tags
    test "$(git rev-parse origin/main)" = "$(git rev-parse fork/main)" && echo "TIPS CONVERGED"
    git grep -c op-nats-changeme fork/main -- swarm/swarm_service.py   # expect: no matches
    git ls-tree fork/main kv-knowledge-packs oracle-memory              # expect: empty
    ```
14. Update local main: `git checkout main && git merge --ff-only origin/main`; delete the integration branch.
15. Log the change in `~/lab/infrastructure/DEPLOY_LOG.md` (house rule) and note the sync in the session handoff.
16. Message Thomas: fork now tracks origin; his experimental content is reachable at tag `fork-pre-sync-2026-08` and canonically in `jlens-experiments`.

**Rollback:** origin's tree is unchanged by the merge, so origin rollback is a no-op concern (worst case: `git push origin 9c3f701:main` fast-forward-rejected -> use a revert commit of the merge, `git revert -m 1`). Fork rollback: `git push fork fork-pre-sync-2026-08:main --force-with-lease` (the one legitimate force case, restoring the tagged pre-sync state).

---

## 9. Risk Assessment

| # | Risk | Sev | Likelihood | Mitigation |
|---|---|---|---|---|
| R1 | `op-nats-changeme` NATS default live on public fork HEAD until merge lands | Med | Certain (present now) | Merge remediates. Session log records NATS rotation 2026-07-17 — **verify rotation** (step 3b); if not rotated, rotate before merging. Default only matters if NATS were exposed off-tailnet; scrub monitors it. |
| R2 | Dual-use injection/steering code public at fork HEAD (contradicts 6a118fa policy); SEP-derived ethics data = licensing exposure | Med | Certain (present now) | Merge removes from HEAD (Option A). **Residual: permanent in both repos' shared pre-base history.** True purge would require coordinated history rewrite of two public repos + GitHub cache/support — explicitly out of scope; accept or open as a separate decision with Thomas/Cara. |
| R3 | Losing fork-unique content via delete-resolution | Low | Ruled out | Verified: fork's `EXPERIMENT_METHODS.md` blob == `9573f26`'s (`36d3d26`); `TRAINING_LAB_NOTEBOOK.md` blobs identical (`5da1d39`). Backstops: preflight step 4 (private-repo check) + step 5 (pre-sync tag). |
| R4 | Commit author email leaks tailnet hostname (`admin@Z420.tail78ae8a.ts.net`) across shared history | Low | Certain, unfixable w/o rewrite | Hostname (not IP); tailnet requires auth to join. Going forward: set clean `user.email` on all machines committing to public repos. |
| R5 | `kintsugi-cma/coverage.xml` committed on both tips with `/home/admin` paths | Low | Certain | Pre-existing on origin too; not merge-blocking. Follow-up commit on origin post-sync: delete + add to `.gitignore`. |
| R6 | Concurrent push moves a tip between preflight and execution | Low | Low | Every step is SHA-pinned with expected values; pushes are non-force (fork push is FF-only, origin push FF-or-PR). Drift causes a loud failure, not silent damage. |
| R7 | Org branch protection squashes/rebases the PR, destroying the dual-parent merge | Med if it happens | Low | Step 11 explicitly requires merge-commit strategy. A squash would leave fork/main un-mergeable-by-FF and re-diverge tips — if it happens, re-merge from the squashed tip (same single conflict). |
| R8 | Thomas expected his fork to keep the experimental suite at HEAD | Med (relationship, not technical) | Possible | Preflight step 3a makes this his explicit call before anything is pushed. Option B documented in Section 6; pre-sync tag preserves his state either way. |

**Net assessment:** technically low-risk — one conflict, lossless resolution, byte-identical result tree, no force-pushes, SHA-pinned procedure with loud-failure checks and a tagged rollback path. The two decisions that must not be made unilaterally are R8 (fork content policy) and R2's residual (history purge or not). Everything else is mechanical.
