# FireGuard Role Hand-off Report

- Member name: (fill in)
- Assigned role: Role 4, Research / Evaluation / Explainability Engineer
- Phase completed: Phase 3 (protocol, engine and baseline-level results DONE; model comparisons NOT RUN: no M1-M4 results exist)
- Date: 2026-11-13
- Branch / PR links: (fill in)

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|
| P3-R4-01 | Freeze P1 before training | **Done** (rule v2, 0 unreadable TIFFs, `status=FROZEN`) | `p1_split_hashes.txt`; `docs/fire_complex_groups.md` 3a |
| P3-R4-02 | Lock comparison protocol | **Done**, before any model result; two operationalisations need lead sign-off | `docs/model_comparison_protocol.md`; `docs/phase_gate_log.md` |
| P3-R4-03 | Ablations and paired comparisons | Engine **done**; run on baselines (reproduces the Phase 2 bootstrap exactly); M1-M4 **NOT RUN** | `evaluation/comparison.py`; `scripts/run_model_comparison.py`; `reports/phase3/model_comparison.md` + CSVs |
| P3-R4-04 | Observability and label quality | Code done, tested on 4 real events; dataset-wide run **pending**; MTBS **not available** | `scripts/run_observability_diagnostic.py`; `reports/phase3/observability_and_label_quality.md` |
| P3-R4-05 | Fold-specific calibration | **Done** for baselines; one-record-per-fold check for model runs; **defect fixed** in `fit_platt` | `reports/phase3/calibration_report.md`, `reliability_tables.csv` |
| P3-R4-06 | Explanations | **UNAVAILABLE** (not retained) | `reports/phase3/explanation_status.md` |

Defect found and fixed: `fit_platt` (plain Newton) diverged to a slope of ~1e18 on the
distance-decay baseline, whose scores underflow to exactly 0 for most cells. Now Newton with
backtracking on the log-likelihood, plus a convergence check that raises instead of returning a
broken calibrator. Regression test reproduces the failure on the old code and passes on the new.
After the fix, the validation-fitted Platt calibrator lowers the decay baseline's test ECE on every
fold (e.g. 2018: 0.00086 -> 0.00019).

## 2. Acceptance tests
| Test | Result | Evidence |
|---|---|---|
| P1 frozen and hash verified before any result | PASS; every runner re-hashes | `run_model_comparison.py`, `run_baselines.py`, Role 3 trainer |
| Identical event/sample/fold definitions | PASS: comparison refuses models whose per-fold events differ | `test_models_must_score_the_same_events` |
| Train-only normalisation, validation-only thresholds/calibration | PASS | `test_normaliser_refuses_non_training_events`, `test_validation_calibration_isolation.py` |
| Full-frame metrics exclude padding | PASS | `test_metrics_masks.py`, Role 3 padding tests |
| Pooled + event-macro + per-event differences | PASS (baselines); models NOT RUN | `ablation_results.csv`, `per_event_differences.csv` |
| Event-group paired bootstrap, documented | PASS | protocol section 3; `test_bootstrap_resamples_whole_groups_not_events` |
| Observability evaluation-only, automated test | PASS (input guard + static scan of model-side code) | `test_observability_not_input.py`, `test_observability_is_evaluation_only.py` |
| Gate 3 decision per module | Rule implemented and tested; decisions NOT RUN | `test_gate3_needs_two_folds_above_zero`; `gate_decisions.json` |
| Negative/null results preserved | PASS by construction (REMOVE rows stay; NOT RUN never imputed) | protocol section 7 |

## 3. Results available now (baselines, frozen P1)
Strongest baseline by validation AUPRC = distance decay on all folds. Decay minus climatology,
per-event AUPRC: +0.0059 [+0.0051, +0.0066], +0.0046 [+0.0034, +0.0057], +0.0053 [+0.0046,
+0.0060], +0.0064 [+0.0056, +0.0073] (2018-2021). 2019 caveat applies.

## 5. Open issues
| Issue | Owner |
|---|---|
| M1 per-event file + `model_version.json` for Gate 2 | Role 2 |
| M3/M4 runs after Gate 2 | Role 3 |
| Dataset-wide observability run (`run_observability_diagnostic.py --data-root`) | Role 4 |
| Lead sign-off of strongest-baseline and Gate 2 operationalisations | lead |
