# FireGuard Role Hand-off Report

- Member name: (fill in)
- Assigned role: Role 1, WSTS / Geospatial Data Engineer
- Phase completed: Phase 3 (data interface DONE; full-dataset release run and HLS audit PENDING)
- Date: 2026-11-13
- Branch / PR links: (fill in; merge after Role 4's and Role 3's branches, whose modules this imports)

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|
| P3-R1-01 | Pinned loader lines, reproduction, 23->40 mapping, target semantics, first scored index | **Done** | `docs/data_dictionary.md` 12.1-12.3; `scripts/verify_against_pinned_loader.py`; `docs/evidence/role1/loader_reproduction.json` |
| P3-R1-02 | Strict input contract | **Done** (`wsts_strict_input_v1`, 7 channels; aspect sin/cos not approved, schema unchanged) | `configs/data/strict_input_v1.yaml`; data dictionary 12.4; angle tests |
| P3-R1-03 | Terrain / land-cover QA | Done on 4 events; **dataset-wide run pending** (one command) | data dictionary 12.5; `docs/evidence/role1/terrain_qa.csv` after the run |
| P3-R1-04 | HLS feasibility | **NOT RUN**; HLS is not an input | `docs/hls_historical_coverage_audit.md` |
| P3-R1-05 | Deterministic full-frame dataset interface | **Done** | `ml/src/fireguard_ml/data/wsts_dataset.py`; `test_wsts_dataset.py` |
| P3-R1-06 | Phase 3 data release | Script done and tested; **full run pending** | `scripts/build_phase3_data_release.py` |

## 2. Acceptance tests
| Test | Command / procedure | Result | Evidence |
|---|---|---|---|
| Pinned revision and source mapping recorded | read of f62dc49 | PASS | data dictionary 12.1 |
| 23->40 mapping verified | pinned loader vs FireGuard on 3 real events, T=1 and T=5 | PASS: 60 samples, all 40 channels equal to 6e-8, 0 target cells differ | `loader_reproduction.json` |
| Strict manifest has only approved channels | config == register == code | PASS | `test_strict_indices_agree_with_the_leakage_register`, `test_strict_input_config_matches_the_code` |
| Exclusions enforced | every excluded channel refused | PASS (17 cases) | `test_every_excluded_channel_is_refused_by_the_register` |
| Aspect schema unchanged; cardinal/nodata/wrap-around behaviour | angle tests | PASS (sin only; convention 0=N, 90=E from the Earth Engine reference) | `test_aspect_*` |
| Input, target, masks share grid after padding | alignment + padding tests; real files | PASS; misaligned or resized frames are refused | `test_misaligned_frames_are_rejected`, `test_padding_keeps_grid_and_marks_padding_invalid` |
| Observability not in inputs | Role 4 guard on sample channels | PASS | `test_evaluation_only_data_cannot_be_offered_as_input` |
| No target leakage | target file read for band 23 only; poisoned other bands never reach x | PASS | `test_target_day_contributes_only_the_target_band` |
| HLS IN MODEL or FEASIBILITY ONLY | - | NOT RUN (not an input) | HLS doc |
| End to end on real data | WSTS source -> Role 3 training (fold 2020) -> OOF export -> replay catalogue | PASS (plumbing only; untrained model) | `docs/evidence/role1/end_to_end_plumbing_check.json` |
| Suite | `cd ml; python -m pytest -q -W error` | 186 passed, 1 skipped (with Roles 3 and 4 merged) | CI |

## 2a. Deviations
| Brief | Done instead | Why |
|---|---|---|
| `docs/channel_mapping.md` | Section 12 of `docs/data_dictionary.md` | Guide asks to update, not duplicate |
| "padding metadata" in the interface | `padding_metadata(hw)` + Role 3's `PaddedBatch.orig_hw` | Padding happens at batching (one place), not per sample |
| Loader NaN handling (`nan_to_num` after year-pair standardisation) | NaN kept; imputed by the fold normaliser with the train-only mean | Same maths, without the loader's statistics that include test years |

## 3. Dataset and source decisions
- Verified: loader semantics (reproduction), aspect convention of the source function, terrain static per event (4 events), both GFS precipitation band names.
- Assumed: VIIRS reflectance scale factor (not needed: channels are standardised per fold).
- Unknown: dataset-wide HHMM 00:xx count, dataset-wide terrain QA, file checksums (all produced by the release run).
- Excluded/leaky: as in the register.

## 4. Interface hand-offs
- Sent to: Role 3 (`WSTSTemporalSource`, registered in `make_source("wsts")`), Role 2 (same source with T=1; use `valid` from the sample, not `make_input_valid_mask`; the register names the mask `binary_active_fire_mask`, Role 2's schema calls it `active_fire_binary`), Role 4 (input manifest), Role 5 (replay catalogue).
- Confirmed by receiver: Role 3 (integration run passes); others pending.

## 5. Open issues and blockers
| Issue | Impact | Owner | Due date |
|---|---|---|---|
| Run `scripts/build_phase3_data_release.py --data-root <WSTS>` on the full dataset and commit its four outputs | Checksums, dataset-wide QA and input manifest | Role 1 | now |
| HLS audit | HLS stays out | Role 1 / Role 6 | if pursued |
| Aspect sin/cos decision | Would make the contract v2 (8 planes) | lead | - |

## 6. Phase sign-off
- Role-specific acceptance: PASS except the dataset-wide release run (pending, one command) and HLS (not run).
- Global gate impact: unblocks Role 2 and Role 3 training on real data once Gate 2 allows it.
