FastAPI, database, FIRMS proxy and forecast artifact retrieval

# FireGuard Backend

The backend provides the API and infrastructure for the FireGuard platform.

## Responsibilities

- REST APIs using FastAPI
- NASA FIRMS server-side integration
- PostgreSQL data storage
- Historical event and observation retrieval
- Precomputed forecast artifact retrieval
- Input validation and error handling

## Architecture

The backend does not perform live PyTorch inference in v1.

Forecast maps are precomputed offline and retrieved through backend APIs.

## Development

Backend dependencies, configuration and startup instructions will be documented here.
