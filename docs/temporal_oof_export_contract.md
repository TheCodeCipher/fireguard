# Temporal OOF export contract (Role 3, P4-R3-02)

Code: `ml/src/fireguard_ml/inference/temporal_oof.py`; CLI: `scripts/export_temporal_oof.py`.
Consumers: Role 4 (evaluation, reveal manifest), Role 5 (registration into the replay backend).

## 1. Which checkpoint predicts which event

For each fold Y, only the fold-Y checkpoint predicts, and only the events whose manifest row for
fold Y says `split = test`. Before writing anything the exporter checks, and raises
`OOFMappingError` otherwise, that:

- the checkpoint is strict (`strict_checkpoint = True`);
- its stored `fold` equals Y and its `split_manifest_hash` equals the current frozen manifest;
- its channel order equals the source's;
- the normaliser file next to it belongs to fold Y;
- every exported event has exactly one fold-Y row and that row is `test`;
- all folds share one `config_hash` (one model version, four fold checkpoints).

## 2. Layout (relative to the artifact root; `artifacts/` is git-ignored)

```
runs/<run_id>/model_version.json
runs/<run_id>/oof_forecasts.jsonl
runs/<run_id>/<event_id>/<origin_date>/probability.tif
runs/<run_id>/post_reveal/per_event_test_scores.csv
```

`origin_date` is the newest input day; the forecast is for `origin_date + 1`.

## 3. `probability.tif`

Single band float32 GeoTIFF, original (unpadded) frame size, the source raster's transform and CRS,
deflate-compressed, NaN on invalid cells. Value = model sigmoid, **uncalibrated**. Tag
`layer=probability`. No uncertainty or explanation layer is produced (none is implemented and
validated). The fold's validation Platt calibrator and F1 threshold are recorded in
`model_version.json` for provenance; they are not applied to the raster.

## 4. `oof_forecasts.jsonl` (pre-reveal; one JSON object per forecast)

Field order is fixed (`PRE_REVEAL_FIELDS`). Mapping to Role 5's models:

| Field | Role 5 target |
|---|---|
| event_id, fold, split_name, is_oof, origin_date, target_date, horizon_hours (24), time_safety_status | ForecastRun |
| protocol (`P1_LOYO`), channel_manifest_sha256 | ForecastRun |
| group_id, split_manifest_sha256, config_hash, channels, height, width, crs, transform, probability_kind, data_contract_version, is_mock | ForecastRun.metadata_json |
| model_version, architecture, checkpoint_sha256 | ModelVersion (name/version, architecture, checkpoint_sha256) |
| artifact_type (`probability_map`), storage_uri, checksum_sha256, content_type (`image/tiff`) | ForecastArtifact (status set by Role 5 on registration) |

Records contain no target path, target value or target-derived metric (`FORBIDDEN_PRE_REVEAL`,
asserted in tests). `channel_manifest_sha256` = SHA-256 of the channel names joined by `\n`.
`time_safety_status = STRICT_LEVEL_A_PASS_CHANNELS_ONLY`. The `architecture` value and the meaning
of `channel_manifest_sha256` need Role 5's confirmation; both are cheap to rename.

## 5. `post_reveal/per_event_test_scores.csv`

Same columns as Role 4's `baseline_per_event_test_scores.csv`
(`fold_test_year, event_id, group_id, baseline, auprc, brier, f1, iou, n_pos, n_valid`); the
`baseline` column carries the run id, so Role 4's paired bootstrap can read both files unchanged.
Pooled test counts per fold are saved next to it as `pooled_counts_fold_<Y>.npz` (scores, pos,
neg, sum_sq, sum_s_pos) so Role 4 can report pooled metrics without the rasters.
F1/IoU use the fold's validation threshold. This file uses the target and must not be exposed
before reveal.

## 6. Mock exports

With `--source synthetic` every record has `is_mock: true` and the CLI refuses a run id without
`mock` in it.
