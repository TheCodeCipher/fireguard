# Reveal contract (Role 4, P4-R4-02) — for Roles 5 and 6

Built by `scripts/validate_oof_and_build_reveal.py` after the OOF run passes Role 4's
independent validation and Role 1's artifact check.

## Pre-reveal (may be served before the reveal action)

`<artifact-root>/reveal_packages/<run_id>/pre_reveal/forecast_sessions.json`, one object per
approved forecast with exactly these fields (`PRE_REVEAL_SESSION_FIELDS`):
forecast_id, event_id, group_id, protocol, fold, origin_date, target_date, horizon_hours,
model_version, architecture, checkpoint_sha256, config_hash, split_manifest_sha256,
data_contract_version, probability_artifact_uri, probability_checksum_sha256, probability_kind,
threshold_fold, threshold, calibrator_fold, fold_validation_auprc, height, width, crs, transform.

`fold_validation_auprc` is the fold's validation score (no information about this forecast's
target). `threshold_fold` and `calibrator_fold` always equal `fold` (checked).

## Behind the reveal action only

`<artifact-root>/reveal_packages/<run_id>/reveal_store/`:
`actual/<forecast_id>.tif` (uint8, 1 = next-day fire, nodata 255, on the forecast's frame;
verified for size, CRS and transform) and `reveal_index.json` with, per forecast_id:
actual_target_uri, actual_target_checksum_sha256, next_day_observability_valid_fraction,
event_metrics_post_reveal (auprc, brier, f1, iou, n_pos, n_valid), threshold, threshold_fold.

The reveal store sits outside `runs/`, so a static file server for forecast rasters cannot reach it.

## Audit record per reveal

`REVEAL_AUDIT_FIELDS`: session_id, event_id, origin_date, forecast_artifact_id,
revealed_target_artifact_id, revealed_at_utc, model_version, fold (`reveal.audit_record`).

## Gaps found in the current backend (main branch), for Role 5

1. `ReplaySession` stores `revealed_at` but not the revealed target artifact id; add
   `revealed_target_artifact_id` (or log `audit_record` rows) to meet the audit requirement.
2. `ACTUAL_TYPES = {"actual_target"}` locks only the target raster. Next-day observability is also
   reveal-only; register it as a locked type (e.g. `next_day_observability`) or keep it inside
   the metrics response, which is already gated.
3. Pre-reveal payload tests: call `fireguard_ml.evaluation.reveal.find_target_leaks(response_json)`
   on the event list, origin list, session, inputs and forecast responses; it must return `[]`.
