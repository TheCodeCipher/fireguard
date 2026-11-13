# FireGuard Role Hand-off Report

- Member name: Arkodyuti Bhattacharyya
- Assigned role: Role 3, Temporal / Weather / GNN Engineer
- Phase completed: Phase 4 — **export tooling only; selection and export BLOCKED**
- Date: 2026-11-13
- Branch / PR links: `role-3/ml/temporal-models`

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|
| P4-R3-01 | Select temporal candidate, freeze metadata | **BLOCKED**: no trained M3/M4, no Gate 3 evidence | selection rule predeclared in `reports/phase3/temporal_ablation_report.md` |
| P4-R3-02 | Export selected OOF predictions | Exporter and CLI **done and tested**; real export **BLOCKED** | `inference/temporal_oof.py`; `scripts/export_temporal_oof.py`; `docs/temporal_oof_export_contract.md` |

## 2. Acceptance tests
| Test | Command / procedure | Result | Evidence |
|---|---|---|---|
| Selection follows Gate 3; nothing unvalidated silently retained | - | NOT RUN (nothing to select). GFS/FiLM/GNN not built, so nothing can be retained by accident | branch report |
| OOF forecasts use the held-out checkpoint for each event | exporter checks fold, split hash, channel order, normaliser fold and the manifest row per event | PASS on MOCK | `test_oof_export_is_fold_safe_and_pre_reveal_clean` (wrong-fold checkpoint raises; every record's manifest row is `test`) |
| Metadata identifies strict vs optional | `time_safety_status`, strict-checkpoint requirement, `is_mock` | PASS (only strict exists) | export contract section 4 |
| No uncertainty/explanation product unless validated | only `probability.tif`, labelled uncalibrated | PASS | export contract section 3 |
| Pre-reveal records carry no target information | field whitelist + forbidden list asserted | PASS | same test |
| CLI dry run on the real frozen manifest (MOCK tensors) | `python scripts/export_temporal_oof.py ... --source synthetic --run-id mock_m4_t3` | PASS: 3,686 forecasts, 210 held-out events, folds 2019/2021 | run log 2026-11-13 |

## 3. Dataset and source decisions
As Phase 3. GeoTIFF transform/CRS come from the sample source; Role 1's loader must provide them.

## 4. Interface hand-offs
- Sent to: Role 5 (field mapping, relative `runs/<run_id>/...` URIs), Role 4 (per-event scores file; reveal manifest stays theirs).
- Confirmed by receiver: pending. Role 5 to confirm `architecture` values and `channel_manifest_sha256` meaning.

## 5. Open issues and blockers
| Issue | Impact | Owner | Due date |
|---|---|---|---|
| Phase 3 blockers (Gate 2, WSTS source) | Nothing to select or export | Roles 1, 2, lead | - |
| Role 5 registration script for `oof_forecasts.jsonl` | Forecasts not visible in replay | Role 5 | - |

## 6. Phase sign-off
- Role-specific acceptance: tooling PASS on MOCK; selection and real export NOT RUN.
- Global gate impact: none yet.
- Recommended next action: after runs A-C and Role 4's paired comparison, pick per the predeclared rule, then `scripts/export_temporal_oof.py --source wsts`.
