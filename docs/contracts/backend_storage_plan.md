# FireGuard Backend Storage Plan — Phase 2

## Principles

- PostgreSQL is the intended shared database; SQLite is the local development/test configuration.
- Store metadata, cache records and provenance in the relational database.
- Keep GeoTIFF/PNG/JSON raster and replay artifacts as files under a configured local/shared storage root.
- No PostGIS, MinIO, TiTiler, Redis, Celery or PyTorch runtime dependency is introduced by this Phase 2 work.
- The backend retrieves precomputed OOF artifacts only. ML training/evaluation generates them offline.

## Logical-to-physical model mapping

| v4 logical concept | Current physical table/model | Notes |
|---|---|---|
| `wsts_events` | `wildfire_events` / `WildfireEvent` | Phase 1 table retained to avoid a destructive rename; exposes only event metadata |
| `model_registry` | `model_versions` / `ModelVersion` | Checkpoint URI/hash and architecture metadata |
| `replay_origins` | `forecast_runs` / `ForecastRun` | Only `is_oof=true` and held-out test split are replayable |
| `oof_predictions` | `forecast_artifacts` / `ForecastArtifact` | Stores type, relative storage URI, checksum and availability status |
| FIRMS observation cache | `fire_observations` / `FireObservationRecord` | Deduplicated metadata records; raw raster/CSV not stored as blobs |
| `replay_metrics` | `replay_metrics` / `ReplayMetric` | Per-run metric values, threshold, aggregation and calibrator ID |
| `data_assets` | `data_assets` / `DataAsset` | Product version, sensing/valid/available times, checksum, URI and QA summary |
| replay reveal state | `replay_sessions` / `ReplaySession` | Enforces forecast-first reveal and session expiry |

## Cache behaviour

A small in-process TTL cache is used for local/single-worker v1. Observation keys include source, bbox, date interval, confidence and minimum FRP. Availability cache keys include source. Cache expiry is configurable. Cache misses can be persisted into `fire_observations`; query cache itself is not persisted. Cache failures are not cached.

A multi-worker production deployment would require a shared cache or sticky routing. That is a later deployment decision, not a reason to add Redis prematurely.

## Artifact convention

`storage_uri` is a path relative to `FIREGUARD_ARTIFACT_ROOT`, for example:

```text
mock/probability_map.json
runs/<run-id>/probability.tif
runs/<run-id>/uncertainty.png
runs/<run-id>/explanation.png
runs/<run-id>/actual_target.tif
```

The local storage adapter rejects absolute paths and traversal. It validates that the file exists before a descriptor or file response is returned. Store SHA-256 and content type alongside each artifact record. Never store raster bytes in PostgreSQL.

## Replay safety

- `ForecastRun.is_oof` must be explicitly true.
- `split_name` must be `test` or `held_out`.
- Input metadata is allow-listed; target URI/next-day validity is not returned.
- A replay session must retrieve the forecast before calling the reveal action.
- Actual-target artifacts and comparison metrics are hidden until that session is marked revealed.
- Session state currently lives in the database; expiry is enforced on every replay request.
- Mock artifacts are synthetic fixtures and must be visibly labelled. They are not model results.

## Deployment caveats

- The migration has been designed for SQLite and PostgreSQL SQLAlchemy dialects, but the current CI/local validation uses SQLite. Actual PostgreSQL integration remains unverified until a PostgreSQL service/instance is available and tested.
- CORS defaults to Vite's local origin. Production origins must be explicitly configured.
- Authentication/authorization and multiple-worker session concurrency hardening are not part of this academic MVP unless the team explicitly adds them.
