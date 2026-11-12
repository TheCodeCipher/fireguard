# FireGuard Role Hand-off Report

- Member name: Arkodyuti Bhattacharyya
- Assigned role: Role 3, Temporal / Weather / GNN Engineer
- Phase completed: Phase 1 (READY FOR REVIEW, not yet PASSED; see section 6)
- Date: 2026-11-11
- Branch / PR links: (fill in)

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|
| P1-R3-01 | Inspect GFS raw ranges and direction encoding | Done | `scripts/audit_gfs_channels.py`; `docs/gfs_channel_audit.md` sections 1-5; `outputs/gfs_channel_range_summary.csv`; `outputs/gfs_wind_direction_histogram.png` |
| P1-R3-02 | Trace GFS issue/valid time and GRIDMET timing evidence | Code trace done; Earth Engine metadata checks not yet run | `docs/gfs_channel_audit.md` sections 6-8; `scripts/ee_checks_gfs_gridmet.js` |
| P1-R3-03 | Document strict temporal input interface | Done (draft, awaiting sign-off) | `docs/temporal_input_contract.md` |
| (follow-up) | GFS precipitation definition change at 2019-11-07 | Script written; result not yet recorded | `scripts/check_gfs_precip_break.py`; audit section 8.4 |

## 2. Acceptance tests
| Test | Command / procedure | Result | Evidence |
|---|---|---|---|
| GFS wind-direction channel identity tied to the WSTS schema | Data dictionary sections 3-4 cross-checked against the export order in `FirePred.py` lines 125-131 | PASS | audit sections 2 and 8.1 |
| Raw min/max/percentiles and histogram saved | `python scripts/audit_gfs_channels.py --data-root <WSTS> --every 10` (1,363 of 13,607 files, 0 skipped) | PASS, provided the two files in `outputs/` are committed | audit section 3 |
| Direction classified | same run | PASS: GFS = `QUADRANT_LIMITED` (min -90, max 90); GRIDMET = `FULL_CIRCLE` | audit section 4 |
| GFS initialization and valid-time evidence documented; unknowns stay UNKNOWN | Code trace of `FirePred.py` lines 60-63 | PARTIAL: run (00Z of day D) is confirmed from code; the lead hours actually selected, Level A and Level B are recorded as UNKNOWN pending EE checks 1-2 | audit sections 6, 8.3, 8.7 |
| Recommendation explicit | n/a | PASS: GFS downwind = DISABLE; GRIDMET downwind = CONDITIONAL; GFS/FiLM = CONDITIONAL | audit section 9; `docs/gfs_branch_gating.md` |
| Role 4 receives the audit evidence before freezing channel decisions | Send the files listed in section 4 | NOT DONE | (add date and acknowledgement) |
| Strict model config contains no GFS channels | `pytest tests/test_temporal_models.py -k strict_config` | PASS | `configs/temporal_unet.yaml` |

## 3. Dataset and source decisions
- Verified:
  - GFS wind direction is stored in [-90, 90]; half of all directions are indistinguishable. Raw u/v are not in the released rasters.
  - GRIDMET wind direction spans 0-360 with all four sectors populated.
  - GFS features come from the 00Z run of UTC day D; frames are UTC calendar days.
  - GFS wind speed is the speed of the mean vector, not the mean speed.
  - The baseline loader applies `sin()` only to angle channels.
- Assumed:
  - GRIDMET `th` is the direction the wind comes from, clockwise from north.
- Unknown:
  - Which GFS lead hours the creation filter really selects (depends on the index format; EE check 1).
  - GFS Level A (borderline at best) and Level B.
  - Why the stored GFS direction is quadrant-limited when the published code uses `atan2`.
  - What the GFS precipitation channel measures on each side of 2019-11-07.
  - Which 24 hours the GRIDMET image for day D covers; GRIDMET publication delay.
- Excluded/leaky:
  - All five GFS channels from strict M1-M4 (project rule).
  - GFS wind direction from every downwind or FiLM use (lossy; final).

## 4. Interface hand-offs
- Sent to: Role 4 (leakage register owner); copy to Role 1 (channel identity) and Role 2 (temporal contract)
- Artifact/schema: `docs/gfs_channel_audit.md` (start at section 8.1, the status register), `outputs/gfs_channel_range_summary.csv`, `outputs/gfs_wind_direction_histogram.png`, `docs/gfs_branch_gating.md`, `docs/temporal_input_contract.md`
- Confirmed by receiver: (pending)

Message for Role 4:

> GFS audit is ready for your register: docs/gfs_channel_audit.md, section 8.1 has one row per finding with a status. Short version: (1) GFS wind direction is QUADRANT_LIMITED, range [-90, 90], u/v not exported, so it is out for downwind and FiLM. (2) GRIDMET direction is full-circle, but I have only assumed the FROM convention and have not checked its day window, so the downwind baseline stays disabled for now. (3) GFS Level A is UNKNOWN, not PASS: F24 is valid at 00:00Z of the target day, and I still need one Earth Engine check to confirm the creation filter selects only F01-F24. (4) GFS precipitation is built by a different rule before and after 2019-11-07. (5) Two things outside GFS for you: the VNP13A1 window can extend past day D, and the PDSI selection depended on the build machine's time zone. I need from you: a ruling on the F24 boundary case, and whether you want sin+cos for GRIDMET direction and aspect before you freeze the strict list.

## 5. Open issues and blockers
| Issue | Impact | Owner | Due date |
|---|---|---|---|
| Run `scripts/ee_checks_gfs_gridmet.js` and fill in audit section 8.7 | Closes GFS lead-hour, Level-A and GRIDMET day-window unknowns | Role 3 (needs Earth Engine access; Role 1 or 4 can run it otherwise) | 2026-11-15 |
| Run `scripts/check_gfs_precip_break.py` and paste the table into audit section 8.4 | Decides whether GFS precipitation can ever be used | Role 3 | 2026-11-15 |
| Record the GRIDMET `th` band description from the Earth Engine catalogue | Needed before a GRIDMET downwind baseline | Role 3 | 2026-11-15 |
| Reviewer name on the audit document | Document is still marked DRAFT | Role 4 | 2026-11-15 |
| `n_leading_observations`, missing-day handling, HDF5 frame order | Blocks final temporal contract | Role 1 / lead | before Phase 2 training |

## 6. Phase sign-off
- Role-specific acceptance: BLOCKED on one item (evidence not yet sent to and acknowledged by Role 4). Everything else is PASS or documented UNKNOWN, which the guide accepts.
- Global gate impact: NONE for G1-C once Role 4 has reviewed. G1-C asks for the direction to be classified and direction-dependent baselines to stay disabled; both hold today. The open EE checks affect the optional GFS/FiLM branch and the GRIDMET downwind baseline, not the strict models.
- Recommended next action: send the hand-off today, run the two outstanding checks, then update sections 8.4 and 8.7 of the audit.
