# GFS, FiLM and downwind branches: gate status (Role 3, task P2-R3-03)

Status date: 2026-11-12. Evidence: `docs/gfs_channel_audit.md`; raw files in `docs/evidence/`.

## Default

M1, M2, M3 and M4 are strict models. They never receive the five GFS forecast channels (GeoTIFF bands 18 to 22, loader processed indices 33 to 37). This is enforced in three places:

| Guard | Where | What it does |
|---|---|---|
| Config test | `ml/src/fireguard_ml/tests/test_temporal_models.py::test_strict_config_has_no_gfs` | Fails if `configs/temporal_unet.yaml` mentions a GFS or forecast channel outside a comment |
| Config consistency test | `ml/src/fireguard_ml/tests/test_temporal_models.py::test_strict_config_is_consistent_with_the_models` | Requires `strict: true` and the five processed indices in `excluded_processed_indices` |
| Checkpoint guard | `fireguard_ml.models.checkpointing.assert_strict_channels` | Refuses to save a strict checkpoint whose channel list contains a name with `gfs` or `forecast` |

The checkpoint guard works on channel names. It relies on Role 4's frozen strict list using the data-dictionary names. An index-based guard belongs with that list once it is frozen.

## Branch status

| Branch | Status today | Blocking reason | What would open it |
|---|---|---|---|
| Downwind baseline with GFS direction | **DISABLED**, final | Stored direction covers only [-90, 90]; half of all directions are indistinguishable and u/v are not in the files (audit sections 4 and 8.2) | Nothing short of rebuilding the channel from raw GFS u/v in Earth Engine, which is out of scope |
| Downwind baseline with GRIDMET direction | **DISABLED**, conditional | Direction is full-circle, but the FROM/TOWARD convention is assumed and the GRIDMET day window is unchecked (audit section 8.5) | EE check 5 recorded; convention confirmed from the catalogue; Role 4 marks GRIDMET direction Level A = PASS |
| GFS / FiLM benchmark-assumption experiment (H5) | **DISABLED**, conditional | Level A for GFS is UNKNOWN: the lead hours the creation filter really selects are unchecked, and F24 sits on the target-day boundary (audit section 8.3) | EE checks 1 and 2 recorded; Role 4 rules Level A = PASS for the channels used |
| GFS direction inside FiLM | **DISABLED**, final | Same loss of direction as the downwind baseline | Nothing |
| GFS precipitation inside FiLM | **DISABLED**, final | Measured: against flat observed precipitation the channel falls from 33.9x to 6.6x observed at the change date, and neither regime is a daily total (audit section 8.4). No rescaling fixes it, because neither regime holds a known quantity. | Nothing short of rebuilding precipitation in Earth Engine from 6-hourly forecast fields, which is out of scope |
| GNN | **NOT STARTED** | Plan v4: only after the main pipeline and baselines are stable | Lead's approval after Gate 2 |

`configs/gfs_benchmark_assumption.yaml` has deliberately not been created. The execution guide allows it only if the audit passes Level A, and it has not.

## If the FiLM branch is opened later

1. It gets its own config file and its own checkpoints saved with `strict=False`; the flag is stored in the checkpoint.
2. At most three GFS channels are candidates: wind speed (band 19), temperature (band 21) and specific humidity (band 22). Direction (20) and precipitation (18) are out on the evidence above. Wind speed is the speed of the 24-field mean vector, which understates speed on days when the wind turns.
3. Results are labelled "benchmark-assumption". No wording about operational availability unless Level B passes separately.
4. As coded upstream, the day-D frame holds the forecast for day D, not for the target day D+1. The write-up must say so.
5. It is compared with the strict model on the same folds, and kept only under the Gate 3 rule. A null result is reported.

## Wind direction in the strict models

GRIDMET wind direction (processed index 7) and aspect (13) reach the model through the baseline loader as `sin(angle)` only, which cannot tell 30 degrees from 150 degrees. Both are full-circle in the raw data, so a sin and cos pair would be valid for them. Whether to add cos is Role 4's and Role 2's decision; it changes `c_strict` and must be made before the strict list is frozen.

Update 2026-11-13: Role 4's register now excludes all GRIDMET channels (Level A UNKNOWN pending EE check 5), so GRIDMET wind direction is not a strict input at all. The strict list is 7 channels; aspect is the only angle channel left, still `sin` only. The sin/cos proposal for aspect is pending and the schema is unchanged until it is approved. Phase 3 status of every branch: `reports/phase3/role3_architecture_and_branch_decisions.md`.
