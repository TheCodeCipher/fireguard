# FireGuard Role 5 Phase 2 implementation overlay

This package contains the Role 5 backend Phase 2 additions for the existing FireGuard repository. It is an overlay, not a replacement for the repository. Apply it from the repository root and preserve the existing Phase 1 migration `a4291d9fbbaa`.

## Included

- `docs/api_contract.md` — human-readable Role 6 API contract.
- `docs/openapi.json` — generated OpenAPI 3 schema for client generation and frontend integration.
- `docs/backend_storage_plan.md` — logical/physical metadata and artifact-storage plan.
- `backend/src/fireguard/` — extended configuration, health/FIRMS/events/replay API, metadata models, cache and local artifact storage.
- `backend/migrations/versions/b6c2d4e91a20_...py` — Phase 2 metadata migration after Phase 1 revision `a4291d9fbbaa`.
- `backend/tests/` — API, cache, FIRMS source/chunking, storage safety and database tests.
- `backend/scripts/` — curated WSTS event catalogue importer, mock replay seed and OpenAPI exporter.

## Apply on Windows PowerShell

From the root of `FireGuard-Repo`, after creating a feature branch:

```powershell
# Extract this ZIP into the repository root, preserving folder paths.
# Then inspect the working tree before staging anything.

git status --short
```

Copy/update the `.env.example` templates, then in `backend/`:

```powershell
python -m pip install -e ".[dev]"
alembic upgrade head
python -m ruff check src tests
python -m pytest tests -v
```

The new migration expects the already-merged Phase 1 migration `a4291d9fbbaa`. Do not delete or rewrite that existing migration.

## Important hand-offs / limits

- Live FIRMS calls require a MAP_KEY held only in backend environment secrets. No key is included.
- FIRMS live network behaviour has not been exercised in this package; HTTP calls are unit-tested with mocked responses.
- The WSTS event importer expects a curated CSV with `event_id` and `year`; map Ananya's actual manifest columns explicitly before import. Do not guess missing metadata.
- Mock replay uses `mock-fire-0001` and synthetic JSON artifacts. It is only for UI/API wiring, not research evaluation.
- This package validates SQLite locally. A real PostgreSQL service test remains required before claiming PostgreSQL deployment validation.
- Role 6 must review and acknowledge the OpenAPI contract and CORS integration before Role 5 Phase 2 is signed off.
