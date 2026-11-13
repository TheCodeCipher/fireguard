# FireGuard Role Hand-off Report

- Member name: Arkodyuti Bhattacharyya
- Assigned role: Role 3, Temporal / Weather / GNN Engineer
- Phase completed: Phase 3 — **pipeline only; training BLOCKED** (cannot be marked complete)
- Date: 2026-11-13
- Branch / PR links: `role-3/ml/temporal-models` (PR: fill in; do not merge before Gate 2)

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|
| P3-R3-01 | Train M3 | **BLOCKED** (pipeline done, not run) | `training/temporal_trainer.py`; `scripts/train_temporal.py`; config |
| P3-R3-02 | Train M4 | **BLOCKED** (pipeline done, not run) | same; M4 trains end to end on MOCK data in tests |
| P3-R3-03 | Temporal-order and sequence ablations | Ordering/offset/no-future checks **done**; T=3 ablation **predeclared, not run** | `training/temporal_data.py::validate_sample`; `reports/phase3/temporal_ablation_report.md` |
| P3-R3-04 | GFS/FiLM stay disabled | **Done** | `reports/phase3/role3_architecture_and_branch_decisions.md`; `docs/gfs_branch_gating.md` |
| P3-R3-05 | GNN | **Deferred** (explicitly) | branch report section 3 |
| P3-R3-06 | Architecture and branch report | **Done** (results sections say NOT RUN) | branch report; `docs/temporal_architecture_note.md` |
| (support) | Temporal sample contract + source interface for Role 1's loader | Done | `docs/temporal_input_contract.md` section 6 |
| (support) | Fold-specific, train-only normalisation | Done | `training/normalization.py` |
| (support) | GPU memory benchmark (needs no data) | Done, **not yet run on the 3060** | `scripts/benchmark_temporal_memory.py` |

## 2. Acceptance tests
| Test | Command / procedure | Result | Evidence |
|---|---|---|---|
| M3/M4 use the same strict inputs and P1 protocol as M1 | config channels == register; trainer verifies FROZEN + manifest hash; samples from `wsts_eval_samples.csv` | PASS (mechanism); M1's own use of the same files is Role 2's to show | `test_temporal_config_lists_exactly_the_registers_strict_channels`, `test_training_refuses_provisional_splits`, `test_training_refuses_a_tampered_manifest` |
| Temporal ordering and target offset match the loader | `validate_sample` on every sample; scored manifest checked: 10,572 samples, 0 gaps | PASS on file dates and on loaded tensors (Role 1 reproduction vs the pinned loader, exact match) | `test_contract_violations_are_rejected`; contract section 5 |
| ConvLSTM forward/backward and output dims | `pytest ml/src/fireguard_ml/tests` | PASS | `test_convlstm_shapes.py`, `test_train_fold_end_to_end_never_touches_test_events[m4_convlstm_unet]` |
| No GFS / GRIDMET / PDSI / NDVI-EVI2 / land cover in strict configs | config tests + `assert_strict_channels` before training | PASS | `test_training_refuses_an_excluded_channel`, `test_strict_config_has_no_gfs` |
| GFS/FiLM and downwind disabled | no config, no code path | PASS | branch report section 2 |
| GNN completed fairly or deferred | - | DEFERRED | branch report section 3 |
| Per-fold configs, seeds, logs, checkpoints, negative results recorded | trainer writes 5 files per fold run | PASS on MOCK runs; **no real runs** | `test_train_fold_end_to_end_*` |
| Normalisation from training groups only | `fit_normalizer` raises on non-train events | PASS | `test_normaliser_refuses_non_training_events` |
| Test split never loaded in training | recording source in test | PASS | `test_train_fold_end_to_end_never_touches_test_events` |
| Full suite | `cd ml; python -m pytest -q -W error; python -m ruff check src` | 136 passed, 1 skipped with Role 4 (186 with Role 1 too) (Role 2 loss-equality, skips until Role 2's module is installed; equality checked by hand against Role 2's branch file on 20 random cases: identical) | CI |

## 2a. Deviations from the Phase 3 brief
| Brief | What I did | Why |
|---|---|---|
| Suggested `configs/models/m3_temporal_unet.yaml` and `m4_convlstm.yaml` | Kept the single `configs/temporal_unet.yaml` (both models + ablation block) | Phase 2 tests already pin that file; two copies of the channel list could drift |
| "Train M3/M4" | Built and tested the training pipeline, including a real-data plumbing run; did not train | Gate 2 not recorded. The WSTS reader is Role 1's `wsts_dataset`; Role 3 has no loader of its own |
| Loss "Role 2's implementation" | Imports Role 2's `masked_bce_dice_loss` if installed, else a verbatim copy | Role 2's code is a separate starter package, not yet merged into `ml/` |

## 3. Dataset and source decisions
- Verified: frame order and target offset (pinned loader code; file dates of all 10,572 scored samples are consecutive); strict list = 7 channels (Role 4's register); largest frame 356 x 311 → padded 384 x 320.
- Verified: Role 1's reader returns processed values in the register's channel order; missing measurements arrive as NaN and the fold normaliser imputes the training mean (contract section 6).
- Unknown: frame order on loaded multi-day tensors and the HDF5 path (Role 1); M4 T=5 peak memory on 6 GB.
- Excluded/leaky: GFS, GRIDMET, PDSI, NDVI/EVI2, year-Y land cover, active_fire_time.

## 4. Interface hand-offs
- Sent to: Role 1 (`TemporalSampleSource` protocol, contract section 6), Role 2 (shared loss, padding semantics), Role 4 (per-event score file in their column format; proposed comparison rules in the ablation report), Role 5 (OOF export contract).
- Artifact/schema: `docs/temporal_input_contract.md`, `docs/temporal_oof_export_contract.md`.
- Confirmed by receiver: pending (all four).

## 5. Open issues and blockers
| Issue | Impact | Owner | Due date |
|---|---|---|---|
| Gate 2 not recorded (M1 vs baselines) | No real M3/M4 training allowed | Role 2 + lead | - |
| ~~No WSTS temporal sample source~~ resolved: `make_source("wsts")` uses Role 1's reader; real-data plumbing run passed | - | Role 1 | done |
| P1 split re-frozen (footprint rule v2, hash `34e29090...`) | Everyone must use the new manifest; the old hash is invalid | Role 4 to commit + notify | now |
| EE check 5 (GRIDMET day window) | GRIDMET stays excluded | **Role 3** | - |
| Comparison rules in ablation report | Must be signed before any run | Role 4 + lead | before training |
| M4 T=5 memory on RTX 3060 | May force a narrower width for both models | Role 3: run `benchmark_temporal_memory.py` | before training |
| Aspect sin/cos | Would change c_strict to 8 | Roles 2, 4, lead | - |

## 6. Phase sign-off
- Role-specific acceptance: mechanism-level criteria PASS; result-level criteria NOT RUN.
- Global gate impact: none on Gate 2; Gate 3 cannot be evaluated for M3/M4 until training runs.
- Recommended next action: run the memory benchmark now; after Gate 2 and the three merges (Role 4, Role 3, Role 1), set `training.enabled: true` and train runs A-C with `--source wsts --data-root <WSTS folder>`.
