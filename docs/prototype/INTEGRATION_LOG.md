# Prototype integration log

Branch `integration/prototype`, created 2026-11-13 from `origin/main` (`0d4d971`). This file lists
every source that went in, every conflict and how it was resolved, every file that was **added**,
**changed** or **removed** during integration, and why. Anything not listed here is exactly as its
owner committed it.

Markers used in code: new files say "Integration" or "INTEGRATION" in their module docstring;
changes inside another role's file are marked with an `INTEGRATION` or `INTEGRATION FIX` comment
at the changed lines.

## 1. Sources

| # | Source | Commit / state | Owner | How it went in |
|---|---|---|---|---|
| 1 | `origin/main` | `0d4d971` (has Phase 2 Role 3 via PR #18 and Phase 2 Role 5 via PR #17) | team | base of the branch |
| 2 | `origin/role-4/research/evaluation` | `a11d7d8` | Role 4 (Sania) | git merge |
| 3 | `role-3/ml/temporal-models` | `5f9b22b` (pushed; same as `origin`) | Role 3 (Arko) | git merge |
| 4 | `origin/role-5/backend/phase2-api-replay` | `17e2d1c` (Phase 3/4 replay) | Role 5 (Tarun) | git merge |
| 5 | `origin/Geospatial-Data-Engineer` | `880ef13` (raster audit, EDA, data dictionary) | Role 1 (Ananya) | git merge |
| 6 | `origin/role2` | `388a9d5` (starter folder + review zip) | Role 2 (Hasan) | git merge, then folded into `ml/` (section 3) |
| 7 | `role-4/` folder | Phase 3/4 Role 4 work prepared in the Role 3 session, built on #2 | Role 4 scope | overlay commit `5cbfc53` |
| 8 | `role-1/` folder | Phase 3/4 Role 1 work prepared in the Role 3 session, built on #5 | Role 1 scope | overlay commit `1349d15` |
| 9 | `FireGuard_Phase1_Phase2_Integrated_Review.zip` (inside #6) | Role 2's newest work, 2026-11-13 | Role 2 | Role 2-owned files copied into `ml/` (commit `c7b5239`) |

Checks before merging: each `branches\*` worktree on disk was compared with its commit; all five
were identical (Role 2's `M` lines were line endings only). All six hashes in
`data/manifests/p1_split_hashes.txt` were re-computed on the files and match, including
`wsts_event_fire_cells.csv.gz`, which Role 2's review had reported as mismatched (their copy had
been re-zipped).

## 2. Merges and conflict resolutions

| Commit | Merge | Conflicts | Resolution |
|---|---|---|---|
| `540121c` | Role 4 branch | none | it already contained `main` |
| `9b975ee` | Role 3 branch | 12 add/add (`temporal_unet.yaml`, GFS docs, Role 3 hand-offs, tests, scripts, `ml/pyproject.toml`) | `main` held exactly Role 3's previous commit `142d428` (squash PR #18), so the branch versions are strictly newer and were taken. `ml/pyproject.toml`: Role 4's dependencies kept and Role 3's `isort known-first-party` pin added |
| `9e2564c` | Role 5 branch | 14 (backend models, routes, schemas, scripts, API docs) | `main` held exactly Role 5's commit `ba78ae7` (squash PR #17); the branch's Phase 3/4 versions are strictly newer and were taken |
| `bdd8b8a` | Role 1 branch | `docs/leakage_audit.csv` add/add | Kept Role 4's decided 20-column register (owner's version; tests read its schema). Role 1's earlier all-PENDING 16-column draft preserved at `docs/archive/leakage_audit_role1_draft_2026-11-05.csv` |
| `9a14963` | Role 2 branch | none | merged as committed |
| `5cbfc53` | Role 4 overlay | n/a (overlay) | Sania's later edit `f9e3b56` (removed two "(fill in)" lines) kept; merged `ml/pyproject.toml` kept instead of the overlay's copy |
| `1349d15` | Role 1 overlay | n/a | `docs/data_dictionary.md` = Ananya's sections 1-11 unchanged + new section 12 |

## 3. Role 2: folded into the canonical package (`c7b5239`)

Role 2's newest work was inside the review zip, already laid out as `ml/src/fireguard_ml/...`, and
Role 2's own README says that layout is canonical and the standalone starter must not be kept.

**Added from Role 2 (unchanged apart from formatting):** `models/unet.py`, `training/losses.py`,
`training/config.py`, `training/smoke_run.py`, `training/real_sample_benchmark.py`,
`data/channel_schema.py`, `data/padding.py`, `data/sample_adapter.py`, `data/temporal_adapter.py`,
`inference/spatial.py`, `ml/tests/test_role2_contracts.py`, `ml/tests/test_spatial_oof_export.py`,
`ml/tests/test_temporal_adapter.py`, `scripts/benchmark_spatial_inference.py`,
`scripts/export_spatial_oof.py`, `configs/unet.yaml`, `ml/constraints-py311-cpu.txt`,
`docs/phase1/*`, `docs/phase3/role2_preparation_status.md`,
`reports/phase3/role2_spatial_robustness.json`; Role 2's additive spatial checkpoint helpers in
`models/checkpointing.py` and the exports in `data/`, `models/`, `inference/__init__.py`.

**Not taken:** Role 2's edit of `configs/temporal_unet.yaml` (it pointed at the withdrawn split
hash `55a3ca18…`); Role 2's exact version pins in `ml/pyproject.toml` (kept as a constraints file).

**Changed:**

| File | Change | Why |
|---|---|---|
| `ml/src/fireguard_ml/data/channel_schema.py` | `PROCESSED_WSTS_CHANNELS` now re-exports Role 1's `PROCESSED_CHANNELS` | 24 of Role 2's 40 hand-written names differed from the loader and the register (`active_fire_binary` vs `binary_active_fire_mask`, `landcover_*` vs `land_cover_*`, `forecast_*` vs `gfs_*`, `pdsi`, `active_fire_detection_time`). Order was identical. Selecting the strict channels by name would have failed |
| `ml/tests/test_role2_contracts.py` | one name updated to `binary_active_fire_mask` | follows the change above |
| `configs/unet.yaml` | split hash `55a3…` -> `34e29090…`; dictionary path fixed; the 7 strict names listed; `strict_approved` still false | withdrawn hash; `docs/data/` existed only in the review copy; same list as Roles 1, 3, 4 |
| Role 2's Python files | `ruff format` only | repo line length is 100 (Role 2 used 120) |
| `ml/pyproject.toml` | `pyyaml` runtime dependency, `scikit-learn` dev dependency, `ml/tests` in `testpaths` | Role 2's config loader needs YAML at runtime; Role 2's metric cross-check; Role 2's tests live in `ml/tests` |
| `.github/workflows/ml-ci.yml` | merged Role 3's (CPU torch, script syntax check) and Role 2's (Python 3.11, lint tests too) | one workflow for one package |
| `ml/README.md` | rewritten: ownership table, install, test | it still said "Expected: 31 passed" and described one role |

**Removed:** `FireGuard_Role2_Starter/` (superseded copy of the same package, with committed
`.pyc` files and `egg-info`) and `FireGuard_Phase1_Phase2_Integrated_Review.zip` (1.4 MB). Both
stay in `origin/role2` history. Role 2's starter docs moved to `docs/role2/`.

## 4. New integration code (`713f8b5`, `791a191`)

| File | Owner area | What it does |
|---|---|---|
| `scripts/make_synthetic_wsts.py` | demo | Writes SYNTHETIC data in the exact WSTS file format (real IDs, dates, frames, CRS, band schema; generated pixels; marker file + GeoTIFF tag) so the whole chain runs without the 40 GB archive |
| `scripts/run_prototype_pipeline.py` | demo | One command: Role 1 release -> Role 3 train (plumbing budget) and export -> Role 1 catalogue -> Role 4 validation/reveal -> bridge. Outputs are always mock |
| `scripts/build_backend_registry.py` | Phase 4 glue | Converts Role 4's reveal package into Role 5's artifact layout, event catalogue, registry bundle and (real mode) Role 5 manifests. Computes per-forecast-day metrics offline with Role 4's metric code. Fail-closed mock/real decision |
| `backend/scripts/load_registry_bundle.py` | Phase 4 glue | Loads a bundle through Role 5's own `import_catalogue` and `register_manifest`; writes `ReplayMetric` rows (no Role 5 tool did) |
| `backend/tests/integration/test_registry_bundle_loader.py` | tests | mock load, real load -> publish -> full reveal flow through the API, Level-A refusal, no target content in any pre-reveal response (reveal contract gap 3) |
| `backend/tests/unit/test_oof_registry_integration_fixes.py` | tests | regression tests for the `horizon_hours` fix |
| `ml/src/fireguard_ml/tests/test_integration_contracts.py` | tests | cross-role agreements: 40 names, strict list in all four configs, data-contract version, split hash, no withdrawn hash, Role 2 vs Role 3 padding identical |
| `frontend/**` | Role 6 | The whole frontend (no Role 6 work existed). See `frontend/README.md` |
| `.github/workflows/frontend-ci.yml` | CI | type-check, tests, build |
| `docs/prototype/*`, `docs/evidence/integration/prototype_demo_run.json`, `docs/prototype/screenshots/*` | docs | this log, runbook, status, demo evidence |

## 5. Changes inside other roles' files

| File | Owner | Change | Why |
|---|---|---|---|
| `backend/src/fireguard/services/oof_registry.py` | Role 5 | `validate_real_oof_context` accepts an integer `horizon_hours` | **Bug:** `register_oof_artifacts.resolve_real_run` requires an int, the validator required a non-empty string (Role 5's unit tests pass `"24"`). Every real manifest failed one of the two checks, so no OOF artifact could ever have been published. Found by running a real-mode bundle through the full CLI path |
| `scripts/build_phase3_data_release.py` | Role 1 | optional `--out-dir` (default behaviour unchanged) | a staged subset or the demo must not overwrite the committed release files |
| `.github/workflows/backend-ci.yml` | Role 5 | also runs on `integration/**` | CI for this branch |
| `.gitignore` | team | `demo_data/`, `prototype_work/` | demo outputs |
| `docs/validation_protocols.md` | Role 4 | `docs/handoffs/` -> `docs/handoff/` in one path | typo, path did not exist |
| `docs/role2/role2_checklist.md` | Role 2 | path to `compute_benchmark.md` | file moved to `docs/role2/` |
| `docs/reveal_contract.md` | Role 4 | status column for the three backend gaps | Role 5 closed gap 1 in Phase 3/4; status recorded |
| `docs/phase_gate_log.md` | Role 4 / lead | rows for integration and Gate 4 (CONDITIONAL) | evidence of this branch |
| `README.md` | team | Prototype section, real repository layout, documentation index, styling row | the layout listed folders that do not exist |

## 6. Decisions made during integration (each reversible, none changes a research result)

1. **One Role 5 model version per fold checkpoint.** Role 3 exports one model version with four
   fold checkpoints; Role 5's `ModelVersion` stores one checkpoint hash. The bridge registers
   `<version>@fold-<Y>`. No schema migration needed. Alternative: add a per-run checkpoint column.
2. **Fold IDs `LOYO-<year>`** in the backend (Role 5's examples), integer years on the ML side.
3. **`STRICT_LEVEL_A_PASS_CHANNELS_ONLY` maps to `PASS_LEVEL_A`.** Same meaning (every input
   channel is Level-A PASS); only this value maps to a publishable status.
4. **Per-forecast-day metrics** are added next to Role 4's event-level metrics; both precomputed
   offline, stored as `ReplayMetric` rows with `aggregation` `forecast_day` / `event`.
5. **Mock is decided at the backend boundary.** Roles 3/4 code runs unchanged on synthetic data
   (that is what makes the demo a real test). The bridge forces mock if the data root is synthetic,
   the run is a plumbing run, `training.enabled` is false, or any record is mock.
6. **Frontend styling is plain CSS**, not Tailwind (plan section 7). Nothing depends on it.
7. **GeoTIFFs are decoded in the browser** (geotiff.js) instead of adding PNG previews to the
   registry, because Role 5's `preview` type is pre-reveal only and could not carry a target.

## 7. Known leftovers (not changed on purpose)

- Role 3's trainer keeps its fallback copy of Role 2's loss (`_role3_masked_bce_dice`); it is only
  used when Role 2's module is missing, which it no longer is (`loss_impl` in run metadata says
  `role2:fireguard_ml.training.losses`).
- Root `README_ROLE1.md`, `README_ROLE4.md` and Role 5's `PHASE3_PHASE4_OVERLAY_README.md` stay
  where their owners put them.
- Historical hand-offs keep the test counts they recorded at the time (31, 143, 223 …).
- `docs/data_dictionary.md` cites `data/manifests/wsts_2018_fire_21458798_active_fire_audit.csv`
  and `docs/validation_protocols.md` cites `data/manifests/p1_loyo_fold_schema.csv`; neither file
  was ever committed by its owner.
- Two committed manifests are over 1 MB: `data/manifests/wsts_raster_grid_audit.csv` (6.8 MB,
  Role 1) and `data/manifests/wsts_event_fire_cells.csv.gz` (1.7 MB, Role 4). Both are compact
  derived manifests their owners committed, not raw data; kept. No raster, checkpoint, database,
  `.env` or key is tracked (checked on the final commit).
