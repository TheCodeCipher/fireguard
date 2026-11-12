# Role 5 Phase 2 — Implementation and validation record

**Status:** Implementation overlay prepared; integration validation remains with the team repository and Role 6.

## Implemented scope

- Versioned FastAPI routes for health, FIRMS availability, FIRMS observation queries, WSTS event metadata, replay origins, replay input metadata, OOF forecast artifact retrieval, metrics, reveal-gated actual artifacts, and static artifact files.
- Exact VIIRS source separation for S-NPP, NOAA-20 and NOAA-21 NRT/SP products.
- Server-side `FIRMS_MAP_KEY`; missing key fails closed with `FIRMS_NOT_CONFIGURED`.
- FIRMS availability check before observation pulls; observation requests are split into at most five-day Area API chunks.
- Input validation, upstream timeouts/errors, source/query-specific cache keys, configurable TTLs, and an explicit disabled-cache state.
- Metadata-only event catalogue with `count` (current page) and `total_count` (matching records).
- OOF-only replay catalogue and retrieval. Actual-target artifacts and replay metrics remain unavailable until the replay session has viewed the forecast and completed the reveal action.
- Local static artifact storage adapter with relative-path and traversal protection.
- SQLAlchemy metadata models and Phase 2 Alembic migration `b6c2d4e91a20`, chained after Phase 1 revision `a4291d9fbbaa`.
- OpenAPI JSON snapshot and human-readable Role 6 API contract.

## Checks performed in the implementation workspace

| Check | Result |
|---|---|
| Pytest | **34 passed** |
| Python compileall (`src`, `tests`, `scripts`) | Passed |
| OpenAPI export | Generated `docs/openapi.json` |
| SQLite migration chain | Upgraded to `b6c2d4e91a20` in a temporary validation tree containing the Phase 1 revision ID and the Phase 2 migration |
| Alembic schema comparison | `No new upgrade operations detected` in that temporary SQLite validation tree |
| Ruff | Not run in this workspace because Ruff could not be installed in the isolated environment; run locally and in GitHub Actions |

## Integration checks still required in the actual FireGuard clone

1. Apply the Phase 2 overlay on a feature branch based on current `main`, preserving Phase 1 migration `a4291d9fbbaa`.
2. Run `alembic upgrade head`, `alembic current`, `alembic check`, Ruff and the complete test suite against the actual Phase 1 repository state.
3. Run a real PostgreSQL migration/connection check when a PostgreSQL service is available. SQLite validation is not proof of PostgreSQL deployment compatibility.
4. Exercise live FIRMS availability and Area API calls with a valid server-side MAP_KEY. Network calls in this package are tested with mocked HTTP responses only.
5. Import Ananya's curated event metadata only after reviewing/mapping the actual CSV fields. Do not invent event dates, footprints, groups or fold IDs.
6. Have Role 6 review and acknowledge `docs/api_contract.md`, generated OpenAPI and CORS behavior.
7. Register real OOF artifacts only after Roles 2 and 4 produce approved files and provenance. The seeded mock artifacts are synthetic UI fixtures, never model results.

## Safety and serving constraints

- The backend has no PyTorch dependency and does not perform inference.
- FIRMS observations remain separate from WSTS predictions.
- No target raster/path is returned in the event catalogue or pre-reveal input metadata.
- Only explicitly `is_oof=true` records in held-out `test`/`held_out` splits are replayable.
- Actual target artifact access requires the matching unexpired replay session and reveal state.
- Raster bytes remain in static files; relational storage keeps metadata and URIs.
