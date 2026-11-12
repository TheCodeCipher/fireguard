# FireGuard API Contract

**Version:** 0.2.0  
**Scope:** Role 5 Phase 2 backend contract for Role 6 frontend integration  
**Status:** Implementation contract; live FIRMS requires a server-side `FIRMS_MAP_KEY`; replay fixtures are synthetic until approved OOF artifacts are registered.

## 1. Product and serving boundaries

FireGuard exposes two distinct data products:

1. **Live Monitor:** NASA FIRMS active-fire observations. These are satellite observations, not FireGuard predictions.
2. **Forecast & Replay Lab:** historical WSTS forecasts retrieved from precomputed out-of-fold (OOF) artifacts. The backend never runs PyTorch, loads a checkpoint, or performs inference.

The API never merges S-NPP, NOAA-20 and NOAA-21 observations into an unlabeled stream. WSTS historical comparison must use the matching `VIIRS_SNPP_SP` source; other products are separately labelled context.

## 2. Base URL and conventions

- Local base: `http://127.0.0.1:8000`
- API prefix: `/api/v1`
- OpenAPI UI: `/docs`
- ReDoc: `/redoc`
- Dates: ISO-8601 `YYYY-MM-DD`.
- Acquisition times: FIRMS UTC `HHMM`.
- Coordinates: decimal degrees, WGS84 longitude/latitude.
- Bounding box: `west,south,east,north`.
- JSON error envelope: `{"error":{"code":"...","message":"..."}}`.
- Pagination: `limit` and `offset` where listed.

## 3. FIRMS source identifiers

Use exactly one source per request:

| API source | Platform label | Product label |
|---|---|---|
| `VIIRS_SNPP_NRT` | `Suomi-NPP` | `NRT` |
| `VIIRS_SNPP_SP` | `Suomi-NPP` | `SP` |
| `VIIRS_NOAA20_NRT` | `NOAA-20` | `NRT` |
| `VIIRS_NOAA20_SP` | `NOAA-20` | `SP` |
| `VIIRS_NOAA21_NRT` | `NOAA-21` | `NRT` |
| `VIIRS_NOAA21_SP` | `NOAA-21` | `SP` |

Do not alias these to a generic `VIIRS` source. Keep source/product/platform visible in marker metadata and map legends.

The MAP_KEY is read only by the backend from `FIRMS_MAP_KEY`. It must not be put in frontend environment variables, query parameters supplied by the browser, logs, or API responses.

## 4. Endpoints

### 4.1 Health

`GET /api/v1/health`

Returns application and database readiness. It does not call FIRMS.

Example:

```json
{
  "status": "healthy",
  "application": "FireGuard API",
  "environment": "development",
  "version": "0.2.0",
  "timestamp": "2026-11-12T10:00:00+00:00",
  "dependencies": {"database": "healthy"}
}
```

A database failure returns `status: degraded` and `dependencies.database: unavailable`.

### 4.2 FIRMS data availability

`GET /api/v1/firms/availability?source=VIIRS_SNPP_SP`

The backend proxies the FIRMS data-availability CSV endpoint and normalizes its minimum and maximum dates. Availability responses are cached per source.

Response fields: `source`, `min_date`, `max_date`, `retrieved_at`, `cache.status`, `cache.ttl_seconds`, `cache.expires_at`.

### 4.3 FIRMS observation query

`GET /api/v1/fires`

Required query parameters:

| Parameter | Type | Rules |
|---|---|---|
| `source` | enum | One exact source from section 3 |
| `start_date` | date | Required, no later than `end_date` |
| `end_date` | date | Required, not in the future |

Optional parameters:

| Parameter | Default | Rules |
|---|---:|---|
| `bbox` | `-180,-90,180,90` | `west,south,east,north`, valid WGS84 bounds, west < east |
| `confidence` | all | `low`, `nominal`, `high`, `unknown` |
| `min_frp` | none | Numeric value >= 0; detections with missing FRP do not pass this filter |
| `limit` | 1000 | 1–5000 |
| `offset` | 0 | >= 0 |

A request may span at most 31 inclusive calendar dates. The server internally divides it into Area API requests of **no more than five days each**. A query is rejected if its requested dates fall outside the source's currently advertised availability.

Example:

```text
GET /api/v1/fires?source=VIIRS_SNPP_SP&start_date=2026-11-01&end_date=2026-11-05&bbox=-125,32,-114,42&confidence=high&limit=1000&offset=0
```

Response (values illustrative; never use these as real detections):

```json
{
  "source": "VIIRS_SNPP_SP",
  "platform": "Suomi-NPP",
  "product": "SP",
  "start_date": "2026-11-01",
  "end_date": "2026-11-05",
  "bbox": "-125,32,-114,42",
  "retrieved_at": "2026-11-12T10:00:00+00:00",
  "data_age_seconds": 1234.0,
  "count": 1,
  "total_count": 1,
  "limit": 1000,
  "offset": 0,
  "has_more": false,
  "cache": {"status": "MISS", "ttl_seconds": 300, "expires_at": "2026-11-12T10:05:00+00:00"},
  "observations": [
    {
      "latitude": 38.1,
      "longitude": -120.2,
      "brightness": 310.0,
      "frp": 8.5,
      "confidence": "high",
      "confidence_raw": "h",
      "platform": "Suomi-NPP",
      "instrument": "VIIRS",
      "acquisition_date": "2026-11-10",
      "acquisition_time": "0945",
      "daynight": "D",
      "source": "VIIRS_SNPP_SP",
      "product": "SP",
      "source_id": "sha256-deduplication-id"
    }
  ]
}
```

`data_age_seconds` is measured from the latest observation's acquisition UTC time at response generation. It is not a guarantee that the source is current. Display `retrieved_at`, source/product, and age in the UI.

### 4.4 Phase 1 mock observations

`GET /api/v1/firms/observations`

This legacy Phase 1 route remains available for offline UI development. It returns a fixed synthetic fixture with `mode: mock`. It is not a NASA proxy and must be visibly labelled as mock data.

### 4.5 WSTS event catalogue

`GET /api/v1/wsts/events?year=2020&limit=100&offset=0`

Optional filters: `year` (2018–2035) and exact `region`; pagination uses `limit` (1–500) and `offset`.

Returns metadata only: event ID, name, year, region, event dates where verified, complex group, P1/P2 fold identifiers, source dataset, and `has_replay`. It returns `count` for the current page and `total_count` for all matching events. It must not return target rasters, target paths, next-day validity, or actual labels.

Events are loaded from the `wildfire_events` table, which is the Phase 1 ORM representation of the v4 `wsts_events` catalogue concept. Empty catalogue returns an empty list; it must not invent WSTS events.

### 4.6 Replay origins

`GET /api/v1/replays/{event_id}/origins`

Only `ForecastRun` records explicitly marked `is_oof=true` and `split_name` equal to `test` or `held_out` are listed. Other runs are never made available as valid replay forecasts.

Response fields include origin date, target date, horizon, protocol, fold ID, model version, split name and OOF flag.

### 4.7 Replay session and prediction-first reveal

A replay session is required. The backend enforces the sequence, not only the frontend.

1. `POST /api/v1/replays/{event_id}/{origin}/session` creates a short-lived session. Response contains `session_id` and `expires_at`.
2. Send `X-Replay-Session-ID: <session_id>` on subsequent calls.
3. `GET /api/v1/replays/{event_id}/{origin}/inputs` returns allow-listed pre-origin input metadata only. Target fields are filtered out.
4. `GET /api/v1/replays/{event_id}/{origin}/forecast` returns descriptors for precomputed probability, uncertainty and explanation artifacts. It marks the forecast as viewed.
5. `POST /api/v1/replays/{event_id}/{origin}/reveal` is allowed only after forecast retrieval.
6. `GET /api/v1/replays/{event_id}/{origin}/actual` is allowed only after reveal.
7. `GET /api/v1/replays/{event_id}/{origin}/metrics` is also held until reveal because metrics summarize comparison with the target.

Artifact descriptors point to `GET /api/v1/artifacts/{artifact_id}`. The same `X-Replay-Session-ID` header is required for artifact retrieval. Actual-target artifacts require the session's reveal state.

Replay artifacts must be static files generated offline. No route in this API launches a model or computes a forecast.

### 4.8 Static artifact retrieval

`GET /api/v1/artifacts/{artifact_id}`

Serves local PNG, GeoTIFF, JSON or other precomputed files through the local storage adapter. `storage_uri` must be relative to the configured artifact root. Absolute paths and `..` traversal are rejected. The database stores metadata/URI/checksum, never raster bytes.

## 5. Cache contract

- Availability cache key: exact FIRMS source.
- Observation cache key: exact source, bbox, start date, end date, confidence, and minimum FRP.
- `limit` and `offset` do not change the underlying observation set and are applied after cache lookup.
- Default availability TTL: 1800 seconds.
- Default observations TTL: 300 seconds.
- Set the relevant TTL to `0` to disable caching; the response reports `cache.status: DISABLED` and no expiry.
- Cache is in-process and suitable for the local/single-worker v1 backend. Multiple workers or hosts would have separate caches; do not claim shared cache consistency. Add Redis only after measured need and team approval.
- Cache errors are not stored.

## 6. Error contract

| HTTP | Code | Meaning |
|---:|---|---|
| 400 | `INVALID_ARTIFACT_URI` | Unsafe or absolute artifact URI |
| 403 | `ACTUAL_NOT_REVEALED` | Actual target/metrics still locked |
| 409 | `REPLAY_NOT_OOF` | Origin is not explicitly held-out OOF |
| 409 | `FORECAST_NOT_VIEWED` | Reveal requested before forecast retrieval |
| 410 | `REPLAY_SESSION_EXPIRED` | Start a new replay session |
| 422 | `VALIDATION_ERROR` / query-specific code | Invalid parameters, bbox or date interval |
| 502 | `FIRMS_UPSTREAM_ERROR` | FIRMS returned an upstream HTTP error or unusable response |
| 503 | `FIRMS_NOT_CONFIGURED` | MAP_KEY is absent from backend environment |
| 503 | `FIRMS_RATE_LIMITED` | FIRMS rate limit was reached |
| 504 | `FIRMS_TIMEOUT` | FIRMS did not respond before timeout |

Do not return raw upstream exception text or URLs, because FIRMS URLs contain the MAP_KEY path segment.

## 7. CORS and browser integration

- Default development origin: `http://localhost:5173`.
- Configure additional exact origins through `FIREGUARD_CORS_ORIGINS` as a JSON array.
- Browser requests use the backend only. The browser must never call the FIRMS URL with a MAP_KEY.
- Send `X-Replay-Session-ID` for replay metadata/artifact calls.

## 8. Data and source boundaries

- FIRMS detections are observations. Do not call them forecasts or FireGuard-generated detections.
- Preserve source/platform/product labels in the response and UI.
- WSTS replay forecasts are western-U.S. benchmark products only and must have a valid OOF fold record.
- Target/actual data is hidden until the reveal endpoint is accepted.
- Next-day VNP09GA observability is not an input/readiness feature in this API.
- HLS, smoke layers, user reports and live model inference are outside this Role 5 Phase 2 implementation.

## 9. OpenAPI

FastAPI serves the current OpenAPI document at `/openapi.json`; a generated snapshot is committed as `docs/openapi.json`. The checked-in API contract is the human-reviewed integration contract. Update both documents and the Role 6 hand-off whenever a public request or response changes.


## 10. Official FIRMS references

- [NASA FIRMS Area API](https://firms.modaps.eosdis.nasa.gov/api/area): supported source identifiers, bbox order, optional date segment and `DAY_RANGE` range (1–5).
- [NASA FIRMS Data Availability API](https://firms.modaps.eosdis.nasa.gov/api/data_availability): per-source availability endpoint and source identifiers.
- [NASA FIRMS API use tutorial](https://firms.modaps.eosdis.nasa.gov/content/academy/data_api/firms_api_use.html): examples for availability and Area API requests.

The implementation intentionally limits each Area API request to at most five days, checks the availability endpoint first, and keeps the MAP_KEY on the backend. Re-check the official documentation if NASA changes the API contract.
