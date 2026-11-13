# FireGuard Role Hand-off Report

- Member name: (fill in)
- Assigned role: Role 4
- Phase completed: Phase 4 (validation and reveal tooling DONE; real OOF run pending)
- Date: 2026-11-13

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|
| P4-R4-01 | Generate and validate OOF predictions | Independent validator **done** (coverage, fold mapping, hashes); generation is Role 3's exporter | `evaluation/oof_validation.py`; `scripts/validate_oof_and_build_reveal.py` |
| P4-R4-02 | Leakage-safe reveal manifest | **Done**: pre-reveal sessions and reveal store in separate trees; audit fields defined | `evaluation/reveal.py`; `docs/reveal_contract.md` |
| P4-R4-03 | Replay metrics and provenance | **Done**: per-event metrics recomputed from rasters + WSTS target; threshold/calibrator fold checked | `recompute_event_metrics`, `compare_metrics`, `thresholds_from_model_version` |

## 2. Acceptance tests
| Test | Result | Evidence |
|---|---|---|
| Every displayed forecast verified OOF for its group | PASS (11 defect types each rejected with a reason) | `test_each_defect_is_rejected_with_its_reason` |
| Targets and observability not exposed pre-reveal | PASS (field whitelist; leak scanner; store outside `runs/`) | `test_leaks_are_detected_anywhere_in_a_payload`, `test_reveal_package_separates_targets_from_pre_reveal` |
| Reveal maps sessions to the correct target artifact | PASS (target raster checked against the forecast frame) | same; real-data run below |
| Fold threshold/calibrator and metric provenance visible | PASS (`threshold_fold`, `calibrator_fold`, hashes in each session) | `docs/reveal_contract.md` |
| OOF coverage and unavailable events reported | PASS (`missing_unavailable` list) | `role4_oof_validation.json` |
| API/browser payload tests | Checker delivered (`find_target_leaks`); Role 5/6 must call it on their responses | reveal contract section "Gaps" |
| Real data, end to end | PASS (plumbing; untrained model): fold 2020, 15 forecasts accepted, 0 rejected, 0 metric mismatches, 15 reveal sessions; coverage of fold 2020 = 15 / 2,861 because only 4 events were available | run of 2026-11-13 |

## 5. Open issues
| Issue | Owner |
|---|---|
| `ReplaySession` lacks `revealed_target_artifact_id`; observability not a locked artifact type | Role 5 |
| Pre-reveal response tests with `find_target_leaks` | Roles 5, 6 |
| Real OOF validation once the selected model is exported | Role 4 |
