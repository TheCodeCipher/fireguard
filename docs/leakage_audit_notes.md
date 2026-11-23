# Leakage audit notes (companion to `docs/leakage_audit.csv`)

- Owner: Role 4 (Research / Evaluation / Leakage Audit)
- Updated: 2026-11-13
- Status: **decisions recorded; awaiting review by Roles 1-3** (Phase 1 acceptance requires their review)

The CSV holds the per-channel evidence. This note explains the decisions, the strict channel list
they produce, and exactly what evidence would change each one. Rules applied (Plan v4 2.3, 4.2):
Level A (no look-ahead) and Level B (operational availability) are independent; `UNKNOWN` is never
treated as `PASS`; a strict-model channel needs Level A `PASS`; Level B only matters for
"operationally available" claims, which FireGuard does not make.

`target_start` = 00:00 UTC on day D+1, because the target is the active-fire map of the UTC
calendar date D+1 (`acq_date` filter in `FirePred.py` lines 125-134).

## 1. Decision table

| Channel(s) | Level A | Level B | Decision | Why, in one line |
|---|---|---|---|---|
| `viirs_m11`, `viirs_i2`, `viirs_i1` (VNP09GA, day D) | PASS | UNKNOWN | **STRICT** | Only images dated D are used, and the daytime overpass is ~20-22 UTC on D, before target start |
| `ndvi`, `evi2` (VNP13A1) | **FAIL** | FAIL | EXCLUDE | A 16-day composite starting after D-15 runs past D into the target window |
| GRIDMET: precipitation, wind speed, wind direction, Tmin, Tmax, ERC, specific humidity | UNKNOWN | **FAIL** | EXCLUDE | Daily windows end at 06/12 UTC, but whether "day D" ends on D or D+1 is unverified; published at least a day late |
| `slope`, `aspect`, `elevation` (SRTM, 2000) | PASS | PASS | **STRICT** | Acquired in 2000, static, cannot contain 2018-2021 fires |
| `palmer_drought_severity_index` | UNKNOWN | UNKNOWN | EXCLUDE | The chosen 5-day image only has to contain D, so it can end up to 4 days later |
| `land_cover_class` (MCD12Q1) | **FAIL** | FAIL | EXCLUDE | Year-Y map is built from all of year Y, including after D and after the fire |
| GFS precipitation | UNKNOWN | UNKNOWN | EXCLUDE (disabled, final) | Meaning changes at 2019-11-07; not a daily total in either period |
| GFS wind direction | UNKNOWN | UNKNOWN | EXCLUDE (disabled, final) | Quadrant-limited to [-90, 90]; u/v not exported |
| GFS wind speed, temperature, specific humidity | UNKNOWN | UNKNOWN | EXCLUDE | Plan v4: no GFS in strict models; Level A also unresolved (F24 = target start) |
| `binary_active_fire_mask` | PASS | UNKNOWN | **STRICT** | Day-D detections only; target is day D+1 |
| `active_fire_time` (HHMM) | PASS | UNKNOWN | EXCLUDE (for now) | Not leakage: the loader's HHMM-to-hour + standardise encoding is forbidden by Plan v4 4.3.7 |

## 2. Resulting strict channel list (7 channels)

`viirs_m11`, `viirs_i2`, `viirs_i1`, `slope`, `aspect`, `elevation`, `binary_active_fire_mask`.

This is enforced in code: `fireguard_ml.data.channel_policy.assert_strict_channels` refuses any
other channel, and `tests/test_no_future_data.py` fails if the register and the reviewed list drift
apart.

**Consequence for the team (needs a decision at Gate 1).** With the evidence available today the
strict models get no weather, no vegetation index, no drought index and no land cover. That is
much thinner than Plan v4's mental picture of M1-M4, and it directly affects:

- M2 / hypothesis H1 (terrain + land cover): land cover as released fails Level A.
- Every model's weather context: GRIDMET is blocked only by one unresolved fact (section 3).

This is the honest output of the rule "unknown is not pass". It is not a reason to relax the rule.

## 3. What would move each blocked channel (owner, evidence needed)

| Channel(s) | Owner | Evidence that would change the decision |
|---|---|---|
| All 7 GRIDMET channels | Role 3 | Earth Engine check 5 in `scripts/ee_checks_gfs_gridmet.js`: if the image for D has `system:time_end` <= D+1 00:00 UTC, Level A becomes PASS and the channels can be STRICT. **Highest-value open item for the whole project.** |
| PDSI | Role 3 / Role 1 | Show the selected image's interval end is <= D for every sample, or re-export with that constraint |
| NDVI, EVI2 | Role 1 (Earth Engine re-export) | Rebuild selecting only composites whose end date is <= D (a lag); that version would pass Level A |
| Land cover | Role 1 (Earth Engine re-export) | Use the year Y-1 MCD12Q1 map; that version would pass Level A |
| `active_fire_time` | Role 1 / Role 2 | Implement minute-of-day sin/cos on fire cells plus a no-detection mask, with a test |
| GFS speed, temperature, humidity | Role 3, then team approval | EE checks 1-2 confirm F01-F24 only; even then GFS is only for the separately labelled benchmark-assumption branch |
| Any Level B | Roles 1/3 | Measured publication timestamps; not needed for any current claim |

## 4. Changes from the previous register version (2026-11-12 -> 2026-11-13)

Evidence text from the earlier version was kept verbatim; each row gained a dated
"Role 4 decision 2026-11-13" sentence. Status changes:

| Channel(s) | Field | Before | After | Evidence added |
|---|---|---|---|---|
| VNP09GA x3 | `no_lookahead` / decision | UNKNOWN / EXCLUDE | PASS / STRICT | `FirePred.py` 107-108 day-D filter + S-NPP daytime overpass time |
| NDVI, EVI2 | A / B | UNKNOWN / UNKNOWN | FAIL / FAIL | `FirePred.py` 109-112 window [D-15, D] + 16-day composite length |
| GRIDMET x7 | B | UNKNOWN | FAIL | gridMET published through "yesterday" (drought.gov); PRISM first run ~24 h after a day ends (prism.oregonstate.edu/calendar) |
| Slope, aspect, elevation | A / B / decision | UNKNOWN / UNKNOWN / EXCLUDE | PASS / PASS / STRICT | SRTM 2000 acquisition (WSTS supplementary) + no date filter (`FirePred.py` 13-14, 86-89) |
| Land cover | A / B | UNKNOWN / UNKNOWN | FAIL / FAIL | `FirePred.py` 97-98 own-year filter + MCD12Q1 annual product uses a full year of observations (user guide) |
| GFS x5 | evidence, creation/valid time | "Role 3 draft" | Role 3 final audit cited; code-traced times recorded | `docs/gfs_channel_audit.md` (final 2026-11-12), `docs/evidence/` |
| Binary active-fire mask | decision | EXCLUDE | STRICT | Existing Level A PASS evidence |

## 5. Things reviewers should challenge

- VNP09GA PASS rests on the overpass-time argument, not on granule timestamps. If Role 1 or 3
  finds the daily image can include acquisitions from D+1, revert to UNKNOWN.
- GRIDMET Level B FAIL rests on documented publication practice, not on WSTS-era timestamps. It
  does not affect any strict decision (those follow Level A).
- Aspect is full-circle (0-360) and must be encoded with sin **and** cos; the baseline loader's
  sin-only transform is not acceptable for FireGuard.
