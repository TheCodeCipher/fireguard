# FireGuard Backend

FastAPI backend for the FireGuard wildfire intelligence platform.

## Requirements

- Python 3.13
- pip
- Git

## Setup

From the backend directory:

```powershell
python -m venv ../.venv
../.venv/Scripts/Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Copy the environment template:

```powershell
Copy-Item .env.example .env
```

## Run the API

```powershell
python -m uvicorn fireguard.main:app --app-dir src --reload
```

API documentation:

- Swagger: http://127.0.0.1:8000/docs
- ReDoc: http://127.0.0.1:8000/redoc

## Available endpoints

| Method | Endpoint                   | Description                   |
| ------ | -------------------------- | ----------------------------- |
| GET    | /                          | API root                      |
| GET    | /api/v1/health             | Health check                  |
| GET    | /api/v1/firms/observations | FIRMS-compatible observations |

The FIRMS endpoint currently uses synthetic mock observations.
It does not retrieve live NASA FIRMS data.

## Database

The current local development configuration uses SQLite.

SQLAlchemy models support the following entities:

- WildfireEvent
- ModelVersion
- ForecastRun
- ForecastArtifact

Alembic manages database schema migrations.

Apply migrations:

```powershell
alembic upgrade head
```

Inspect current migration:

```powershell
alembic current
```

Generate a new migration:

```powershell
alembic revision --autogenerate -m "describe schema change"
```

## Testing

Run all tests:

```powershell
python -m pytest tests -v
```

Run linting:

```powershell
python -m ruff check src tests
```

## Architecture notes

- Historical forecasts are retrieved from precomputed artifacts.
- Model inference is not performed inside the API.
- Large raster artifacts are stored externally; the database stores metadata and storage references.
- FIRMS observations and historical forecasts are separate data products.
- Mock observations must never be presented as real satellite detections.

## Current limitations

- Live FIRMS integration is not implemented.
- PostgreSQL deployment has not yet been validated.
- Production authentication and authorization are not implemented.
- Forecast artifact retrieval endpoints are planned for a later phase.
