# Phase 4 OOF Artifact Manifest Contract

**Version:** 1.0  
**Owner:** Role 5 — Backend / MLOps  
**Purpose:** Bind precomputed static artifacts to one frozen, held-out OOF forecast run.

## 1. Manifest structure

A real manifest binds to an event already present in the imported WSTS catalogue. It supplies model/run metadata so the registry can create the corresponding `ModelVersion` and held-out `ForecastRun` rows, or optionally verify an existing `forecast_run_id`. Values are not inferred from filenames.

```json
{
  "manifest_version": "1.0",
  "mode": "real",
  "event_key": "event-21458798",
  "model_name": "FireGuard M1 strict U-Net",
  "model_version": "1.0.0",
  "architecture": "UNet2D",
  "checkpoint_sha256": "<64-hex-sha256>",
  "config_hash": "<64-hex-sha256>",
  "protocol_id": "P1_LOYO",
  "fold_id": "LOYO-2020",
  "split_name": "test",
  "is_oof": true,
  "origin_date": "2018-01-06",
  "target_date": "2018-01-07",
  "horizon_hours": 24,
  "held_out_group_id": "complex-group-001",
  "split_manifest_sha256": "<64-hex-sha256>",
  "input_manifest_sha256": "<64-hex-sha256>",
  "data_schema_version": "strict-input-v1",
  "data_schema_sha256": "<64-hex-sha256>",
  "channel_manifest_sha256": "<64-hex-sha256>",
  "time_safety_status": "PASS_LEVEL_A",
  "artifacts": [
    {
      "artifact_type": "probability_map",
      "storage_uri": "p1_loyo/unet-v1/LOYO-2020/event-21458798/2018-01-06/probability.tif",
      "content_type": "image/tiff",
      "checksum_sha256": "<64-hex-sha256>",
      "file_size_bytes": 123456,
      "reveal_class": "pre_reveal",
      "metadata": {
        "format": "GeoTIFF",
        "crs": "EPSG:32610",
        "affine_transform": [375, 0, 500000, 0, -375, 4500000],
        "width": 224,
        "height": 288,
        "nodata": null,
        "valid_extent": [0, 0, 224, 288],
        "units": "probability"
      }
    },
    {
      "artifact_type": "actual_target",
      "storage_uri": "restricted/p1_loyo/unet-v1/LOYO-2020/event-21458798/2018-01-06/actual_target.tif",
      "content_type": "image/tiff",
      "checksum_sha256": "<64-hex-sha256>",
      "file_size_bytes": 654321,
      "reveal_class": "post_reveal",
      "metadata": {
        "format": "GeoTIFF",
        "crs": "EPSG:32610",
        "affine_transform": [375, 0, 500000, 0, -375, 4500000],
        "width": 224,
        "height": 288,
        "nodata": null,
        "valid_extent": [0, 0, 224, 288],
        "units": "binary_active_fire_detection"
      }
    }
  ]
}
```

All hashes and numeric values above are placeholders illustrating structure, not actual FireGuard evidence.

## 2. Valid artifact types and reveal classes

| `artifact_type` | Required `reveal_class` | Namespace | Raster metadata |
|---|---|---|---:|
| `probability_map` | `pre_reveal` | normal | Yes |
| `preview` | `pre_reveal` | normal | Yes |
| `uncertainty_map` | `pre_reveal` | normal | Yes |
| `explanation_map` | `pre_reveal` | normal | Yes |
| `provenance` | `pre_reveal` | normal | No |
| `actual_target` | `post_reveal` | `restricted/` | Yes |
| `metrics` | `post_reveal` | `restricted/` | No |

A real manifest must contain a `probability_map` and an `actual_target` before it can be published. Optional layers may be omitted. Mock manifests set `mode="mock"`, each artifact metadata object sets `mode="mock"`, and they can never be published.

## 3. Validation rules

- Manifest version must be `1.0`.
- `forecast_run_id` must resolve to one existing OOF run with `split_name` `test` or `held_out`.
- Event key, complex group, origin/target dates, protocol and fold must match the database row.
- Protocol must be `P1_LOYO` or `P2_GEO_HOLDOUT` for real runs.
- Checkpoint, config, split, input, data-schema and channel-manifest hashes must be valid SHA-256 values and match their registered model/run values where present.
- Both run and manifest must state `time_safety_status="PASS_LEVEL_A"`.
- Artifact paths must be relative, root-contained POSIX paths; no traversal, backslashes or symlink escape.
- Checksum and file size are recomputed from bytes.
- Raster width/height are positive integers; affine transform has six finite numeric coefficients; valid extent is within pixel bounds.
- A wrong reveal class, missing file, checksum mismatch, unapproved OOF mapping or invalid schema prevents validation/publication.

## 4. CLI state transition

```text
manifest + files
      |
      v
   generated
      |
      | all metadata, OOF, path and checksum checks pass
      v
   validated
      |
      | explicit --publish after gate approval
      v
   published
```

`mock` is a separate non-publishable state. `revoked` is hidden from replay. Existing Phase 2 `available` records become `generated` during the Phase 3 migration and must be revalidated.
