# FireGuard Role Hand-off Report

- Member name: (fill in)
- Assigned role: Role 4, Research / Evaluation / Leakage Audit Engineer
- Phase completed: Phase 1 (READY FOR REVIEW; one run left before the manifests can be frozen, see section 6)
- Branch: `role-4/ml/evaluation-protocol` (suggested, per CONTRIBUTING.md)
- Date: 2026-11-13
- Branch / PR links: (fill in)

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|
| P1-R4-01 | WSTS paper and official code review | Done (earlier) | `docs/validation_protocols.md` sections 1-13 |
| P1-R4-02/03 | Two-level leakage register, every channel decided | Done; review by Roles 1-3 pending | `docs/leakage_audit.csv`, `docs/leakage_audit_notes.md` |
| P1-R4-04 | Coordinate channel/time evidence with Roles 1 and 3 | Done for the evidence received (Role 3 final GFS audit; Role 1 grid audit); open items listed in section 5 | `docs/validation_protocols.md` section 16.1 |
| P1-R4-05 | Related-fire grouping rule, diagnostics, 20 % check, 50 km / 30 d sensitivity | Rule, code and tests done; run on real metadata (provisional) | `docs/fire_complex_groups.md`, `docs/fire_complex_groups_report.md`, `fire_complex_groups.py`, `derive_event_footprints.py` |
| P1-R4-06 | P1 defined separately from P2 | Done; P2 explicitly deferred | `docs/validation_protocols.md` sections 14-15 |
| P1-R4-07 | Whole-group validation from training years only, 15-20 % | Done (17.5-17.7 % in every fold) | `event_splits.py`, section 14 |
| P1-R4-08 | Freeze and hash manifests | **Built and hashed as PROVISIONAL**; freezing needs the footprint run (section 6) | `data/manifests/p1_loyo_fold_manifest.csv`, `p1_split_hashes.txt` |
| P1-R4-09 | Metric definitions and code | Done, implemented in full, not stubs | `docs/metrics_contract.md`, `evaluation/metrics.py`, `bootstrap.py`, `calibration.py` |
| P1-R4-10/11 | Soft baselines specified; downwind baseline status | Done; downwind disabled | `docs/evaluation_protocol.md`, `baselines/downwind.py` |
| P1-R4-12 | Observability evaluation-only, with a guard test | Done | `data/observability.py`, `test_observability_not_input.py` |
| P1-R4-13 | Threshold, early stopping and calibration on validation only | Done, enforced in code | `evaluation/calibration.py`, `test_validation_calibration_isolation.py` |

## 2. Acceptance tests
| Test | Command / procedure | Result | Evidence |
|---|---|---|---|
| Every strict-model candidate has Level-A and Level-B status with evidence; unknown strict channels excluded | `pytest -k "leakage or no_future"` | PASS (pending Roles 1-3 review) | `docs/leakage_audit_notes.md` |
| Grouping report has method, thresholds, size distribution, largest-group share by year | `python scripts/build_groups_and_p1_splits.py` | PASS on provisional input | `docs/fire_complex_groups_report.md` |
| > 20 % warning evaluated; fallback documented | same | PASS: max share 4.1 % (6.5 % under 50 km / 30 d), single linkage kept; complete-linkage fallback implemented and tested | report; `test_fire_complex_groups.py` |
| P1 splits group-safe; validation only from training years | `pytest -k event_splits` | PASS | `test_event_splits.py` |
| P1 manifests frozen and hashed; no group across splits | same + hash check | **BLOCKED**: hashed and valid, but `status=PROVISIONAL` until real footprints are used | `p1_split_hashes.txt` |
| P2 documented separately or deferred | n/a | PASS (deferred) | `validation_protocols.md` section 15 |
| VNP09GA next-day observability not an input; test prevents it | `pytest -k observability` | PASS | `test_observability_not_input.py` |
| Metric definitions, full-frame natural prevalence, event-level paired bootstrap documented | n/a | PASS | `docs/metrics_contract.md` |
| Role 1-3 channel/time evidence reflected accurately | Review | PASS for evidence received | register rows dated 2026-11-13 |
| Whole ML suite | `cd ml`, then `pytest` and `ruff check src` | 108 passed (31 existing + 77 new); ruff clean | CI `ml-ci.yml` |

## 2a. Deviations from the brief, stated for the reviewer
| Brief says | Done instead | Why |
|---|---|---|
| `src/fireguard/...`, top-level `tests/` | `ml/src/fireguard_ml/...`, `ml/src/fireguard_ml/tests/` | `backend/` owns the `fireguard` name; the ML suite lives inside its package (same as Role 3) |
| Footprints from Role 1 | Role 4 derives active-fire cell footprints from the TIFFs | Role 1 marks perimeters UNKNOWN and footprints NOT_DERIVED; frame bounding boxes are image extents, not footprints |
| Anchor year = calendar year of earliest start date | Manifest year of the earliest-starting event | Avoids a non-existent 2017 fold for `fire_21458836` (starts 2017-12-29, manifest year 2018) |
| Metric "stubs" (`validation_protocols.md` 13.5) | Full implementations, cross-checked against scikit-learn and a pixel-level recomputation on real data | Phase 2 needs them anyway |
| Hash file bytes | Hash with CRLF normalised to LF | `.gitattributes` does not pin CSVs to LF; a Windows checkout would otherwise break the hash |

## 3. Dataset and source decisions
- Verified: 607 events / 13,607 TIFFs; consecutive dates in every event; all readable TIFFs EPSG:32610, 375 m, 23 float32 bands, no nodata tag; strict channel list of 7 (`viirs_m11`, `viirs_i2`, `viirs_i1`, `slope`, `aspect`, `elevation`, `binary_active_fire_mask`).
- Assumed: VNP09GA daily images contain only acquisitions from calendar day D (overpass-time argument, not granule timestamps).
- Unknown: GRIDMET day-window alignment (Level A); PDSI interval end; GFS lead hours; Level B for most channels.
- Excluded/leaky: NDVI/EVI2 (Level A FAIL), land cover (Level A FAIL), all GFS (strict rule; precipitation and direction disabled outright), GRIDMET and PDSI (Level A unknown), HHMM (encoding).

## 4. Interface hand-offs
- Sent to: Roles 1, 2, 3 (register and splits), Role 2 (metrics contract, evaluation sample list, per-event results format)
- Artifact/schema: `docs/leakage_audit.csv` + notes; `data/manifests/fire_complex_groups.csv`; `p1_loyo_fold_manifest.csv`; `wsts_eval_samples.csv`; `p1_split_hashes.txt`; `docs/metrics_contract.md` section 7 (per-event CSV format)
- Received from Role 3: final GFS audit and evidence (2026-11-12). Confirmed received: yes.
- Received from Role 1: event metadata, spatial metadata, per-TIFF grid audit. Confirmed received: yes.
- Confirmed by receivers: (pending)

## 5. Open issues and blockers
| Issue | Impact | Owner | Due date |
|---|---|---|---|
| Run footprints + rebuild manifests on a machine with the dataset (two commands, ~15 min) | Blocks G1-D (frozen P1 manifests) | Role 4 | 2026-11-15 |
| Strict list has no weather, vegetation or land cover | Team must accept a thin M1-M4 or resolve the blockers below | Team at Gate 1 | 2026-11-15 |
| GRIDMET Level A (Earth Engine check 5) | Would add 7 weather channels to strict | Role 3 | ASAP |
| Re-export NDVI/EVI2 with composite end <= D, land cover from year Y-1 | Would restore vegetation and land cover (M2 / H1) | Role 1 | Phase 2 |
| 20 TIFFs unreadable in Role 1's grid audit (Google Drive path) | Footprint completeness; re-copy and confirm | Role 1 | 2026-11-15 |
| Pinned loader commit / local patches reconciliation | Source-review closure | Role 1 | Phase 2 |
| HHMM encoding (sin/cos + mask) | `active_fire_time` stays excluded until done | Role 1 / Role 2 | Phase 2 |

## 6. Phase sign-off
- Role-specific acceptance: **BLOCKED** on one item: P1 manifests are valid and hashed but `PROVISIONAL`. Unblock: `python scripts/derive_event_footprints.py --data-root <WSTS folder>` then `python scripts/build_groups_and_p1_splits.py`; the header of `p1_split_hashes.txt` must then read `status=FROZEN`. Everything else PASS, subject to Roles 1-3 review of the register.
- Global gate impact: **BLOCKING for G1-D** until the freeze run; G1-B (leakage register) ready for review; G1-C reviewed (Role 3's GFS classification is reflected in the register).
