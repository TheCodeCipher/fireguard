# FireGuard Role Hand-off Report

- Member name: Arkodyuti Bhattacharyya
- Assigned role: Role 3, Temporal / Weather / GNN Engineer
- Phase completed: Phase 1 (READY FOR REVIEW, not yet PASSED; see section 6)
- Branch: `role3/ml/temporal-models`
- Date: 2026-11-12 (final)
- Branch / PR links: (fill in)

## 1. Tasks completed
| Task ID | Description | Status | Evidence/artifact |
|---|---|---|---|
| P1-R3-01 | Inspect GFS raw ranges and direction encoding | Done | `scripts/audit_gfs_channels.py`; `docs/gfs_channel_audit.md` sections 1-5; `outputs/gfs_channel_range_summary.csv`; `outputs/gfs_wind_direction_histogram.png` |
| P1-R3-02 | Trace GFS issue/valid time and GRIDMET timing evidence | Code trace done; Earth Engine metadata checks scripted but not yet run, so the affected items stay UNKNOWN | `docs/gfs_channel_audit.md` sections 6-8; `scripts/ee_checks_gfs_gridmet.js` |
| P1-R3-03 | Document strict temporal input interface | Done (draft, awaiting sign-off) | `docs/temporal_input_contract.md` |
| (follow-up) | GFS precipitation definition change at 2019-11-07 | Done. Break confirmed against observed precipitation; channel recommended DISABLE | `scripts/check_gfs_precip_break.py`; audit section 8.4; `docs/evidence/gfs_precip_break_check.csv` |

## 2. Acceptance tests
| Test | Command / procedure | Result | Evidence |
|---|---|---|---|
| GFS wind-direction channel identity tied to the WSTS schema | Data dictionary sections 3-4 cross-checked against the export order in `FirePred.py` lines 125-131 | PASS | audit sections 2 and 8.1 |
| Raw min/max/percentiles and histogram saved | `python scripts/audit_gfs_channels.py --data-root <WSTS> --every 10` (1,363 of 13,607 files, 0 skipped) | PASS. `outputs/` is git-ignored, so committed copies are in `docs/evidence/` | audit section 3; `docs/evidence/` |
| Direction classified | same run | PASS: GFS = `QUADRANT_LIMITED` (min -90, max 90); GRIDMET = `FULL_CIRCLE` | audit section 4 |
| GFS initialization and valid-time evidence documented; unknowns stay UNKNOWN | Code trace of `FirePred.py` lines 60-63 | PARTIAL: run (00Z of day D) is confirmed from code; the lead hours actually selected, Level A and Level B are recorded as UNKNOWN pending EE checks 1-2 | audit sections 6, 8.3, 8.7 |
| Recommendation explicit | n/a | PASS: GFS downwind = DISABLE; GRIDMET downwind = CONDITIONAL; GFS/FiLM = CONDITIONAL | audit section 9; `docs/gfs_branch_gating.md` |
| Role 4 receives the audit evidence before freezing channel decisions | Send the files listed in section 4 | **NOT DONE — the only unmet item in this table** | (add date and acknowledgement) |
| Strict model config contains no GFS channels | `cd ml && pytest -k strict_config` | PASS | `configs/temporal_unet.yaml` |

## 2a. Deviations from the Phase 1 brief, stated for the reviewer

| Brief says | What I did | Why |
|---|---|---|
| "Inspect the raw HDF5 channel arrays" | Audited the released GeoTIFFs (13,607 files, 1,363 sampled) | The HDF5 is a conversion of these same rasters, so the values are the same and the GeoTIFFs carry the band order the creation code writes. Flag it if the team wants the HDF5 path checked separately. |
| "Work from actual channel indices/names supplied by Role 1" | Used `docs/wsts_data_dictionary.md` and cross-checked against the export order in `FirePred.py` lines 125-131 | Role 1 had not issued a channel manifest when this ran. The two sources agree, but this is an unconfirmed dependency: Role 1 should confirm the indices in section 2 of the audit. |
| Commit `outputs/gfs_wind_direction_histogram.png` and `outputs/gfs_channel_range_summary.csv` | Committed copies under `docs/evidence/` | The repository's `.gitignore` excludes `outputs/`. The guide and the `.gitignore` contradict each other; this resolves it without changing either. |
| "representative unique/sample values" | Covered by the percentile table plus the histogram reading in audit section 8.2 (the distribution is folded, not clipped) | A unique-value listing is not meaningful for a continuous float channel; the clipped-versus-folded question is what that check exists to answer. |

## 3. Dataset and source decisions
- Verified:
  - GFS wind direction is stored in [-90, 90]; half of all directions are indistinguishable. Raw u/v are not in the released rasters.
  - The GFS precipitation channel changes meaning at 2019-11-07: measured against flat observed precipitation it falls from 33.9x to 6.6x observed. Neither regime is a daily total.
  - GRIDMET wind direction spans 0-360 with all four sectors populated.
  - GFS features come from the 00Z run of UTC day D; frames are UTC calendar days.
  - GFS wind speed is the speed of the mean vector, not the mean speed.
  - The baseline loader applies `sin()` only to angle channels.
  - The GFS direction channel is folded rather than clipped: the histogram has no spike at the two extremes (audit section 8.2).
- Assumed:
  - GRIDMET `th` is the direction the wind comes from, clockwise from north.
- Unknown:
  - Which GFS lead hours the creation filter really selects (depends on the index format; EE check 1).
  - What quantity the precipitation channel actually holds in each regime, and why 2020 and 2021 differ by a factor of 11 although both lie after the change date.
  - GFS Level A (borderline at best) and Level B.
  - Why the stored GFS direction is quadrant-limited when the published code uses `atan2`.
  - Which 24 hours the GRIDMET image for day D covers; GRIDMET publication delay.
- Excluded/leaky:
  - All five GFS channels from strict M1-M4 (project rule).
  - GFS wind direction from every downwind or FiLM use (lossy; final).

## 4. Interface hand-offs
- Sent to: Role 4 (leakage register owner); copy to Role 1 (channel identity) and Role 2 (temporal contract)
- Artifact/schema: `docs/gfs_channel_audit.md` (start at section 8.1, the status register), `outputs/gfs_channel_range_summary.csv`, `outputs/gfs_wind_direction_histogram.png`, `docs/gfs_branch_gating.md`, `docs/temporal_input_contract.md`
- Confirmed by receiver: (pending)

Message for Role 4:

> GFS audit is ready for your register: docs/gfs_channel_audit.md, section 8.1 has one row per finding with a status, and the raw evidence is in docs/evidence/. Short version: (1) GFS wind direction is QUADRANT_LIMITED, range [-90, 90], u/v not exported, so it is out for downwind and FiLM. (2) GRIDMET direction is full-circle, but I have only assumed the FROM convention and have not checked its day window, so the downwind baseline stays disabled for now. (3) GFS Level A is UNKNOWN, not PASS: F24 is valid at 00:00Z of the target day, and I still need one Earth Engine check to confirm the creation filter selects only F01-F24. (4) GFS precipitation changes meaning at 2019-11-07 and I have measured it: against observed GRIDMET precipitation, which is flat across the two periods at 0.49 mm/day, the channel falls from 33.9x observed to 6.6x. Neither period is a daily total, so I am recommending DISABLE rather than a per-period correction. (5) Two things outside GFS for you: the VNP13A1 window can extend past day D, and the PDSI selection depended on the build machine's time zone. I need from you: a ruling on the F24 boundary case, and whether you want sin+cos for GRIDMET direction and aspect before you freeze the strict list.

## 5. Open issues and blockers
| Issue | Impact | Owner | Due date |
|---|---|---|---|
| Run `scripts/ee_checks_gfs_gridmet.js` and fill in audit section 8.7 | Closes GFS lead-hour, Level-A and GRIDMET day-window unknowns | Role 3 (needs Earth Engine access; Role 1 or 4 can run it otherwise) | 2026-11-15 |
| Record the GRIDMET `th` band description from the Earth Engine catalogue | Needed before a GRIDMET downwind baseline | Role 3 | 2026-11-15 |
| Reviewer name on the audit document | Document is still marked DRAFT | Role 4 | 2026-11-15 |
| Branch is named `role3/gfs-audit`; CONTRIBUTING.md requires `role-3/ml/<task>` | Breaks the repo's naming rule and the backend CI branch filter | Role 3 | before opening the PR |
| `n_leading_observations`, missing-day handling, HDF5 frame order | Blocks final temporal contract | Role 1 / lead | before Phase 2 training |

## 6. Phase sign-off
- Role-specific acceptance: BLOCKED on one item (evidence not yet sent to and acknowledged by Role 4). Everything else is PASS or documented UNKNOWN, which the guide accepts.
- Global gate impact: NONE for G1-C once Role 4 has reviewed. G1-C asks for the direction to be classified and direction-dependent baselines to stay disabled; both hold today. The open EE checks affect the optional GFS/FiLM branch and the GRIDMET downwind baseline, not the strict models.
- Recommended next action: send the hand-off today, run the two outstanding checks, then update sections 8.4 and 8.7 of the audit.
