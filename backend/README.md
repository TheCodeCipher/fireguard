# FireGuard Backend

FastAPI backend for FireGuard. The API proxies FIRMS observations and retrieves static, precomputed OOF artifacts for historical replay. It does not run PyTorch inference.

## Requirements

- Python 3.11 or newer
- PostgreSQL for a shared deployment, or SQLite for local development/tests
- A NASA FIRMS MAP_KEY only for live FIRMS proxy routes

## Setup (PowerShell, from `backend/`)

```powershell
# Activate the repository-level venv if already created
..\.venv\Scripts\Activate.ps1

python -m pip install -e ".[dev]"
Copy-Item .env.example .env
```

Set `FIRMS_MAP_KEY` in the local backend `.env` only when using live FIRMS. Never commit `.env`. The browser never receives this key.

## Database

Apply schema migrations:

```powershell
alembic upgrade head
alembic current
```

Local default is SQLite. The migrations target SQLAlchemy-supported SQLite/PostgreSQL types, but PostgreSQL connectivity must be verified against an actual PostgreSQL service before describing deployment as tested.

## Run the API

```powershell
python -m uvicorn fireguard.main:app --app-dir src --reload
```

- Swagger: http://127.0.0.1:8000/docs
- OpenAPI JSON: http://127.0.0.1:8000/openapi.json
- Health: http://127.0.0.1:8000/api/v1/health

## Main Phase 2 endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/v1/health` | API and database health |
| GET | `/api/v1/firms/availability` | Server-side FIRMS data availability proxy |
| GET | `/api/v1/fires` | FIRMS observations by source, date and bbox |
| GET | `/api/v1/firms/observations` | Stable Phase 1 synthetic mock fixture |
| GET | `/api/v1/wsts/events` | Metadata-only WSTS event catalogue |
| GET | `/api/v1/replays/{event_id}/origins` | Held-out OOF replay origins only |
| POST | `/api/v1/replays/{event_id}/{origin}/session` | Start replay session |
| GET | `/api/v1/replays/{event_id}/{origin}/inputs` | Pre-origin input metadata only |
| GET | `/api/v1/replays/{event_id}/{origin}/forecast` | Retrieve registered static OOF forecast artifacts |
| POST | `/api/v1/replays/{event_id}/{origin}/reveal` | Unlock the actual target after forecast retrieval |
| GET | `/api/v1/replays/{event_id}/{origin}/actual` | Actual artifact descriptor after reveal |
| GET | `/api/v1/artifacts/{artifact_id}` | Static local artifact retrieval |

Full request/response and error contract: `../docs/api_contract.md`.

## Mock replay fixture

After applying migrations, from `backend/`:

```powershell
python scripts/seed_mock_replay.py
```

This creates event `mock-fire-0001` and a tiny synthetic forecast/actual grid under `artifacts/mock/`. It is deliberately labelled as mock data. Never report it as a WSTS event or model result.

## Tests and lint

```powershell
python -m ruff check src tests
python -m pytest tests -v
```

All tests use synthetic fixtures and mocked HTTP responses; CI does not need a FIRMS key, WSTS archive or GPU.

## Environment variables

See `.env.example`. Important values:

- `DATABASE_URL`
- `FIRMS_MAP_KEY` (server-side only)
- `FIREGUARD_FIRMS_BASE_URL`
- `FIREGUARD_FIRMS_TIMEOUT_SECONDS`
- `FIREGUARD_FIRMS_OBSERVATION_CACHE_TTL_SECONDS`
- `FIREGUARD_FIRMS_AVAILABILITY_CACHE_TTL_SECONDS`
- `FIREGUARD_ARTIFACT_ROOT`
- `FIREGUARD_REPLAY_SESSION_TTL_SECONDS`
- `FIREGUARD_CORS_ORIGINS` (JSON array)

## Serving boundaries and limitations

- FIRMS endpoints return observations, not forecasts.
- Source IDs for S-NPP, NOAA-20 and NOAA-21 remain distinct.
- Area API requests are chunked into windows of no more than five days.
- The in-process TTL cache is for local/single-worker use; multiple workers have independent caches.
- Replay routes serve static files only. No PyTorch dependency or model inference path exists in the backend runtime.
- The replay session state is database-backed. User authentication/authorization is not implemented in this academic MVP.
- Do not register a replay run unless `is_oof=true` and its split is explicitly `test` or `held_out`.
- Actual targets and target-derived metrics stay unavailable until the replay session's reveal action is accepted.
- SQLite has been used for local validation; PostgreSQL requires a separate integration run.


## Phase 3/4 — Offline OOF artifact registry

The backend is retrieve-only: train/infer/export outside FastAPI, then register static OOF artifacts. Use `scripts/register_oof_artifacts.py` from this directory to validate and register the JSON manifest. Run `--dry-run` first; omit `--publish` to stop at `validated`. Only publish after the frozen split, checkpoint/config hashes and Gate 3/OOF evidence are approved.

See `../docs/backend_artifact_storage.md` and `../docs/interfaces/phase4_oof_artifact_contract.md`. Never commit real rasters, model checkpoints, raw datasets or the `artifacts/restricted/` target namespace.

## Loading a pipeline registry bundle (added during the Prototype integration)

`scripts/load_registry_bundle.py` loads what `../scripts/build_backend_registry.py` produces from
Role 4's reveal package. It does not bypass anything above: events go through
`import_wsts_event_catalogue.import_catalogue`, artifacts through
`register_oof_artifacts.register_manifest` (all validation and lifecycle rules), and it also
writes the precomputed `replay_metrics` rows that the metrics endpoint serves after reveal.

```powershell
python -m alembic upgrade head
python scripts/load_registry_bundle.py --bundle <registry folder>/registry_bundle.json
python scripts/load_registry_bundle.py --bundle <registry folder>/registry_bundle.json --publish
```

Mock bundles (synthetic data, plumbing runs, or anything before Gate 2) load only with
`FIREGUARD_APP_ENV=development` and can never be published. Set `FIREGUARD_ARTIFACT_ROOT` to the
same folder the bundle's files were copied into. See `../docs/prototype/RUNBOOK.md`.

Integration fix in this area: `services/oof_registry.validate_real_oof_context` now accepts the
integer `horizon_hours` that `register_oof_artifacts.py` requires (before, every real manifest
failed one of the two checks). Regression tests: `tests/unit/test_oof_registry_integration_fixes.py`.
