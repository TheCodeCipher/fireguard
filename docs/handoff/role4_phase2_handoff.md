# FireGuard Role Hand-off Report

- Member name: (fill in)
- Assigned role: Role 4, Research / Evaluation / Explainability Engineer
- Phase completed: Phase 2 (CODE COMPLETE; results pending the baseline run on frozen manifests, see section 6)
- Branch: `role-4/ml/evaluation-protocol` (suggested)
- Date: 2026-11-13
- Branch / PR links: (fill in)

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|
| P2-R4-01 | Frozen manifests load exactly; isolation tests before any result | Done: the runner validates the manifest and refuses to run on a hash mismatch | `scripts/run_baselines.py`, `test_event_splits.py` |
| P2-R4-02 | Binary persistence and dilated persistence (F1/IoU) | Done | `baselines/persistence.py` |
| P2-R4-03 | Continuous distance-to-fire decay, parameters on validation | Done (scale by validation Brier, amplitude in closed form) | `baselines/distance_decay.py` |
| P2-R4-04 | Training-fold distance-bin climatology | Done, training groups only | same |
| P2-R4-05 | Downwind-weighted decay | Disabled with reason | `baselines/downwind.py` |
| P2-R4-06 | Logistic regression / boosting | Not done (optional) | n/a |
| P2-R4-07 | AUPRC, F1, IoU, Brier, BSS, reliability with masks | Done | `evaluation/metrics.py`, `test_metrics_masks.py` |
| P2-R4-08 | Pooled and event-macro; paired bootstrap over groups | Done | `evaluation/bootstrap.py` |
| P2-R4-09 | Per-fold thresholds/calibrators, validation only | Done | `evaluation/calibration.py` |
| P2-R4-10 | Validation result tables | Done (generated report) | `docs/baseline_results.md` (after the run) |
| P2-R4-11 | Observability report interface + guard test | Done | `data/observability.py` |
| P2-R4-12 | Per-year results with 2019 caveat | Built into the report | `run_baselines.py` |
| P2-R4-13 | M1 outputs tied to exact manifests and config hash | Interface defined | `docs/metrics_contract.md` section 7 |

## 2. Acceptance tests
| Test | Command / procedure | Result | Evidence |
|---|---|---|---|
| P1 manifest used unchanged by every experiment | Runner hash check; `test_committed_p1_manifest_is_valid_and_unchanged` | PASS | `run_baselines.py`, `test_event_splits.py` |
| Binary baselines with F1/IoU; soft baselines give continuous scores | `pytest -k baselines` | PASS | `test_baselines.py` |
| Climatology fitted on training only | same | PASS (fitting on validation/test raises) | `test_climatology_cannot_be_fitted_outside_training` |
| Thresholds/calibrators only on validation, natural prevalence | `pytest -k isolation` | PASS | `test_validation_calibration_isolation.py` |
| Test never influences threshold, parameter or calibrator | same + code review | PASS by construction | runner selects on `validation` rows only |
| Full-frame metrics ignore padding/invalid cells | `pytest -k masks` | PASS (adversarial padding test) | `test_metrics_masks.py` |
| Pooled and event-macro reported | runner | PASS | `docs/baseline_results.md` layout |
| Bootstrap unit is event group | `pytest -k bootstrap` | PASS (test fails if events are resampled individually) | `test_bootstrap_resamples_whole_groups_not_events` |
| Observability evaluation-only and blocked from inputs | `pytest -k observability` | PASS | `test_observability_not_input.py` |
| Per-year/per-fold tables with 2019 caveat | runner output | PASS (layout); real numbers pending | report template |
| Real-data check | 3 real events staged from the archive, run end to end and recomputed pixel by pixel with scikit-learn | PASS: AP 0.050434, Brier 0.006913, dilated F1 0.110833 identical both ways | session log |

## 2a. Deviations from the brief, stated for the reviewer
| Brief says | Done instead | Why |
|---|---|---|
| Distance decay scale fitted on train/validation | Scale chosen by validation **Brier**, not AUPRC; amplitude added | AUPRC is identical for every scale (monotone transform); without an amplitude every burning cell scores 1.0 although only ~10 % reburn next day, which wrecked Brier (BSS -1.97 vs +0.03 after the fix on the real-data check) |
| Binary baselines "for F1/IoU" | No AUPRC is reported for them at all | A 0/1 map has one operating point; its AUPRC is degenerate |
| Pixel-level metric arrays | Counts grouped by distinct score; streaming histogram for model outputs | A test fold is ~300 M cells; exact pooling without holding pixels in memory |

## 3. Dataset and source decisions
- Verified: baseline inputs use only the day-D active-fire band (Level A PASS); target is day D+1 binarised as in the loader.
- Assumed: none beyond Phase 1.
- Unknown: none new.
- Excluded/leaky: downwind decay (no valid direction).

## 4. Interface hand-offs
- Sent to: Role 2 (M1 comparison), Roles 3 (M3/M4 comparison)
- Artifact/schema: `configs/evaluation.yaml`; `docs/metrics_contract.md` section 7 per-event CSV; `docs/evidence/role4/baseline_per_event_test_scores.csv` (after the run) as the paired-bootstrap reference
- Confirmed by receiver: (pending)

## 5. Open issues and blockers
| Issue | Impact | Owner | Due date |
|---|---|---|---|
| Run `scripts/run_baselines.py` after the P1 manifests are frozen | No baseline numbers until then | Role 4 | End of Week 3 |
| Role 2 must emit per-event scores in the contract format | Needed for the Gate 2 paired bootstrap | Role 2 | Gate 2 |
| Optional logistic regression / boosting baseline | Optional | Role 4 | if time |

## 6. Phase sign-off
- Role-specific acceptance: **CODE COMPLETE, RESULTS PENDING**. Every acceptance test passes on synthetic data and on a three-event real-data check; the per-fold results table is produced by one command once the manifests are frozen: `python scripts/run_baselines.py --data-root <WSTS folder>`.
- Global gate impact: CARRY-OVER into Gate 2 (Role 2's M1 comparison needs `docs/baseline_results.md`).
