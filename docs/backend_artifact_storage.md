# FireGuard Backend Artifact Registry and Storage

**Owner:** Role 5 — Backend / MLOps  
**Phase:** 3 and 4  
**Status:** Implementation contract; real OOF artifacts remain unpublished until the frozen split/model metadata and files pass validation.

## 1. Serving boundary

The backend is a **static artifact registry and retrieval service**. It does not import PyTorch, load model checkpoints, select a model, run preprocessing, or perform inference. ML jobs generate files offline. Role 5 validates their manifests and checksums, records metadata, and serves the selected static files.

The registry extends the existing `forecast_runs` and `forecast_artifacts` tables. It does not introduce a parallel artifact table.

## 2. Artifact lifecycle

| State | Meaning | Replay visible? |
|---|---|---:|
| `generated` | File/metadata was produced but has not passed registry validation | No |
| `validated` | Required OOF context, file, checksum and metadata passed validation | No |
| `published` | Validated real OOF artifact explicitly published for replay | Yes |
| `mock` | Explicit synthetic fixture; visible only in development | Development only |
| `revoked` | Removed from replay after integrity/provenance issue | No |

Allowed real flow: source file generated offline → first registry operation validates it and writes `validated` → a separate explicit publish operation changes that unchanged record to `published`. The CLI refuses to publish an artifact that has not already been registered as `validated`. Revalidation cannot demote `published`; `revoked` is terminal. Mock records cannot be published. Existing Phase 2 `available` rows are migrated to `generated`, because their registry-level OOF validation has not been established.

## 3. Storage layout

All paths are relative to `FIREGUARD_ARTIFACT_ROOT` (default `backend/artifacts/`). Example:

```text
backend/artifacts/
├── p1_loyo/
│   └── unet-v1/
│       └── LOYO-2020/
│           └── event-21458798/
│               └── 2018-01-06/
│                   ├── probability.tif
│                   ├── probability_preview.png
│                   ├── uncertainty.tif            # optional
│                   ├── explanation_gradcam.tif    # optional
│                   └── provenance.json
├── restricted/
│   └── p1_loyo/
│       └── unet-v1/
│           └── LOYO-2020/
│               └── event-21458798/
│                   └── 2018-01-06/
│                       ├── actual_target.tif
│                       └── metrics.json
├── mock/
│   ├── probability_map.json
│   └── actual_target.json
└── registry/
    └── manifests/                 # compact JSON manifests only
```

Rules:

- Every real post-reveal file (`actual_target` and target-derived `metrics`) must live below `restricted/`.
- Real pre-reveal files must not live below `restricted/`.
- Mock files use `mock/` and have `metadata.mode = "mock"`.
- Use only relative POSIX paths; absolute paths, `..`, backslashes and symlink escapes are rejected.
- Never commit real raster files, WSTS data, HDF5, checkpoints, or the restricted target namespace to Git. Commit only code, documentation, and small synthetic fixtures.

## 4. Canonical registry fields

The canonical registry is the existing relational data plus `ForecastArtifact.metadata_json`:

- `ForecastRun`: event, origin/target date, horizon, protocol, fold, split name, OOF flag, channel-manifest hash, time-safety status and run metadata.
- `WildfireEvent`: event key and frozen complex-group ID.
- `ModelVersion`: model name/version/architecture, checkpoint SHA-256 and config hash.
- `ForecastArtifact`: artifact type, safe relative storage URI, checksum, content type, lifecycle status and artifact metadata.
- `ReplaySession`: forecast-viewed/revealed timestamps and bounded audit-event JSON.

For a real published run, the registration manifest must bind exactly to an existing `is_oof=true` `test`/`held_out` run, the event's `complex_group_id`, protocol/fold, origin/target dates, model checkpoint/config hashes, split manifest hash, input manifest hash, data schema version/hash, channel manifest hash and `PASS_LEVEL_A` time-safety status. Unknown or mismatched values fail closed.

Raster artifacts require `format`, `crs`, six affine coefficients, width, height, explicit nodata value (which may be JSON null), and `valid_extent` in pixel coordinates `[xmin, ymin, xmax, ymax]`.

## 5. Registration CLI

The command runs from `backend/` and uses the configured database and artifact root:

```powershell
# Validate a manifest and all referenced files without DB changes
python scripts/register_oof_artifacts.py --manifest path\to\oof_manifest.json --dry-run

# Register as validated, but not yet visible to replay
python scripts/register_oof_artifacts.py --manifest path\to\oof_manifest.json

# Explicit publication after the model/evaluation owner and team lead approve Gate 3/OOF evidence
python scripts/register_oof_artifacts.py --manifest path\to\oof_manifest.json --publish
```

The CLI does not run inference. It verifies paths, SHA-256, file size, raster metadata, run/fold/group identity, model checkpoint/config hashes, protocol, split, schema and Level-A status. A successful non-publish run writes `validated`; `--publish` writes `published` only if the manifest also includes both `probability_map` and `actual_target`. Mock manifests require development mode and remain `mock` permanently.

## 6. Integrity and serving

- The registry computes SHA-256 and file size from bytes; supplied metadata cannot override the computed values.
- The serving path is resolved under the artifact root on every access. The file is re-hashed before it is served; missing, modified, size-mismatched or unsafe files fail closed.
- A descriptor exposes artifact ID, type, content type, checksum and an allow-listed display/geospatial metadata subset. It never exposes a filesystem path or the actual-target URI.
- Artifact requests require `X-Replay-Session-ID`. Pre-reveal artifacts require that the session loaded the forecast. Post-reveal artifacts additionally require that the session completed the reveal transition.
- Post-reveal responses use `Cache-Control: private, no-store` and `X-Content-Type-Options: nosniff`.
- The reveal transition is recorded in `ReplaySession.audit_json` with timestamp, artifact IDs, protocol and fold. Repeated reveal calls are idempotent.

The replay session is a product-flow gate for the research prototype, not a substitute for authentication/authorization in a public production deployment.

## 7. Backend/ML runtime separation

The backend's runtime dependency list intentionally excludes `torch`, `torchvision`, CUDA and model-serving libraries. Offline training/inference and export run in the ML environment. FastAPI only reads validated metadata and static files.
