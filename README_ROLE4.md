# Role 4 work package: Phase 1 and Phase 2

Everything in this folder uses the **same paths as the `fireguard` repository**. To add it to the
repo, copy the folder's contents (except this README) into the repository root on a Role 4 branch,
for example `role-4/ml/evaluation-protocol`.

## Phase 3 and Phase 4 (added 2026-11-13)

New: comparison engine and gates (`evaluation/comparison.py`, `scripts/run_model_comparison.py`),
locked protocol (`docs/model_comparison_protocol.md`), calibration report and a `fit_platt` fix,
evaluation-only observability diagnostic, explanation status (UNAVAILABLE), independent OOF
validation and the reveal package (`evaluation/oof_validation.py`, `evaluation/reveal.py`,
`scripts/validate_oof_and_build_reveal.py`, `docs/reveal_contract.md`), gate log, hand-offs.
Results in `reports/phase3/` are baseline-only; every model row says NOT RUN.

One command still needs the dataset (10-15 minutes):
`python scripts/run_observability_diagnostic.py --data-root <WSTS folder>`; commit
`reports/phase3/observability_strata.csv`. Do **not** commit
`data/evaluation_only/next_day_observability.csv` unless the team agrees (it is target-day data;
keeping it out of the repo is the simplest guard). Phase 4 commands run after the real OOF export.

## Update 2026-11-13: re-frozen split and a baseline fix (read first)

1. **The P1 split was re-frozen.** The first frozen grouping over-grouped (one 64-event group,
   399 test events in 2018) because persistent thermal anomalies and unrelated fires in the same
   frame created false footprint overlaps. Footprint rule v2 removes cells seen in events from 3+
   different years and keeps only each event's own fire. New manifest sha256:
   `34e2909009ad3bb7f5aefb3ad9b5c02ec1df4c776eb2c5bf02eaf68f098e6976`. Details and evidence:
   `docs/fire_complex_groups.md` section 3a. **Tell Role 1 and anyone who received the old hash.**
2. **Baseline bug fixed.** A rounding error made the dilated baseline identical to persistence
   and left the nearest climatology bin empty. Fixed in `baselines/distance_decay.py`, with
   regression tests; `docs/baseline_results.md` regenerated on the v2 split.
3. Re-run check: `cd ml; python -m pytest -q` (116 tests with the Role 4 files), then
   `python scripts/run_baselines.py --data-root <WSTS folder>` only if you want to reproduce the
   numbers (they are already in the files).

The paragraph below describes the state before the freeze and is kept for history.

## Status in one paragraph

All Role 4 Phase 1 and Phase 2 code, tests and documents are done (108 tests pass with the
existing suite; ruff clean). Every channel in the leakage register now has a decision backed by
named evidence. The P1 manifests are built, validated and hashed, but marked **PROVISIONAL**,
because the fire-complex grouping needs real fire footprints, which require the extracted dataset.
Three commands on a machine with the extracted dataset turn them into **FROZEN** manifests and
produce the baseline results (section 1).

## 1. Finish the run (needs the extracted WildfireSpreadTS folder; about 30 minutes)

Run in PowerShell, one line at a time, from this folder:

```powershell
python -m pip install scipy pyproj pyyaml rasterio
python scripts\derive_event_footprints.py --data-root ..\WildfireSpreadTS\WildfireSpreadTS
python scripts\build_groups_and_p1_splits.py
python scripts\run_baselines.py --data-root ..\WildfireSpreadTS\WildfireSpreadTS
```

Check afterwards:

- line 2 of `data\manifests\p1_split_hashes.txt` says `status=FROZEN`;
- `docs\fire_complex_groups_report.md` no longer says PROVISIONAL, and no year's largest-group
  share is above 20 %;
- `docs\baseline_results.md` exists and says FROZEN;
- the footprint step printed `"tiffs_unreadable": 0`. If not, re-copy the listed TIFFs and
  repeat steps 2-4.

New files from the run: `data\manifests\wsts_event_footprints.csv`,
`data\manifests\wsts_event_fire_cells.csv.gz`, `docs\baseline_results.md`,
`docs\evidence\role4\baseline_results.json`, `docs\evidence\role4\baseline_per_event_test_scores.csv`.
A cache is written to `outputs\role4\` (git-ignored).

## 2. Read these first, in this order

1. `docs/handoff/role4_phase1_handoff.md`: what is done, what is blocked, Gate 1 impact.
2. `docs/leakage_audit_notes.md`: every channel decision in plain terms. **Only 7 channels are
   strict-safe today** (three VIIRS reflectance bands, slope, aspect, elevation, the binary fire
   mask). No weather, vegetation or land cover until the blockers in its section 3 are resolved.
   This needs a team decision at Gate 1.
3. `docs/fire_complex_groups.md`: the grouping method, plus answers to Role 1's seven questions.
4. `docs/metrics_contract.md` and `docs/evaluation_protocol.md`: what every other role must follow.
5. `docs/handoff/role4_phase2_handoff.md`.

## 3. Judgement calls the owner should review before sending

These are evidence-backed, but they are Role 4 decisions and the owner must be able to defend them:

| Decision | Where the reasoning is |
|---|---|
| VNP09GA reflectance (day D) Level A **PASS** (overpass-time argument) | `leakage_audit_notes.md` sections 1 and 5 |
| NDVI/EVI2 Level A **FAIL** (16-day composite runs past D) | same, section 1 |
| Land cover Level A **FAIL** (annual map uses the whole year) | same, section 1 |
| GRIDMET Level B **FAIL** (published at least a day late) | same, section 4 |
| Anchor year = manifest year, not calendar year of first date | `fire_complex_groups.md` section 3 |
| Decay baseline scale chosen by Brier, with a fitted amplitude | `evaluation_protocol.md` |

## 4. Files

| Path | New / changed | What it is |
|---|---|---|
| `docs/leakage_audit.csv` | changed (her file) | Every row decided; earlier evidence kept, a dated decision appended |
| `docs/validation_protocols.md` | changed (her file) | Sections 14-16 added: P1 protocol, P2 deferral, status update; 0-13 unchanged |
| `docs/leakage_audit_notes.md` | new | Decisions, strict list, what would change each one |
| `docs/fire_complex_groups.md` | new | Grouping method |
| `docs/fire_complex_groups_report.md` | new, generated | Grouping diagnostics and fold sizes |
| `docs/metrics_contract.md`, `docs/evaluation_protocol.md` | new | Metric definitions and protocol |
| `docs/handoff/role4_phase1_handoff.md`, `role4_phase2_handoff.md` | new | Hand-off reports |
| `docs/evidence/role4/grouping_and_split_diagnostics.json` | new, generated | Machine-readable diagnostics |
| `data/manifests/fire_complex_groups.csv`, `p1_loyo_fold_manifest.csv`, `wsts_eval_samples.csv`, `p1_split_hashes.txt` | new, generated | Groups, P1 folds, scored samples, hashes |
| `data/manifests/wsts_event_metadata.csv` | **Role 1's file**, byte-identical copy | Needed by the scripts. Commit it only if Role 1's branch is not merged yet |
| `configs/evaluation.yaml` | new | Evaluation settings |
| `ml/src/fireguard_ml/data/`: `fire_complex_groups.py`, `event_splits.py`, `channel_policy.py`, `observability.py` | new | Grouping, splits, register enforcement, observability guard |
| `ml/src/fireguard_ml/evaluation/`: `metrics.py`, `bootstrap.py`, `calibration.py` | new | Metrics, group bootstrap, per-fold thresholds and calibration |
| `ml/src/fireguard_ml/baselines/` | new | Persistence, distance decay, climatology, downwind (disabled) |
| `ml/src/fireguard_ml/tests/conftest.py` + 8 `test_*.py` | new | 77 tests |
| `scripts/derive_event_footprints.py`, `build_groups_and_p1_splits.py`, `run_baselines.py` | new | The three commands in section 1 |
| `ml/pyproject.toml` | **changed (shared)** | Adds `scipy` to dependencies and `pyproj`, `pyyaml` to the `scripts` extra |
| `__init__.py` files | unchanged (already in the repo, empty) | Included so this folder runs on its own |

## 5. Tests

From the repository root after copying the files in:

```powershell
cd ml
pip install -e ".[dev]"
pytest
ruff check src
```

Expected: 108 passed (31 existing + 77 Role 4), ruff clean. One test cross-checks AUPRC against
scikit-learn and is skipped if scikit-learn is not installed (it is not a dependency).
