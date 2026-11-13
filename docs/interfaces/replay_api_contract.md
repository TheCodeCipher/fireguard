# FireGuard Replay API Contract — Phase 4

**Owner:** Role 5 — Backend / MLOps
**Version:** 1.0
**Boundary:** retrieve-only static OOF replay. No model inference in FastAPI.

Every request after session creation sends `X-Replay-Session-ID`. All origins are backed by OOF runs and a published probability artifact. Synthetic mock artifacts are visible only when `FIREGUARD_APP_ENV=development` and are always identified as mock.

## Flow

1. `GET /api/v1/wsts/events` — metadata-only event catalogue; `has_replay` is true only if a published OOF probability artifact exists (or a clearly marked mock exists in development).
2. `GET /api/v1/replays/{event_id}/origins` — returns only origins with published OOF probability artifacts. Includes `registry_status` and `is_mock`.
3. `POST /api/v1/replays/{event_id}/{origin}/session` — starts a TTL-bounded session.
4. `GET /api/v1/replays/{event_id}/{origin}/inputs` — allow-listed pre-origin metadata only.
5. `GET /api/v1/replays/{event_id}/{origin}/forecast` — returns pre-reveal artifact descriptors and a safe provenance object; marks the session forecast-viewed.
6. `GET /api/v1/artifacts/{artifact_id}` — static file retrieval. The forecast must have been loaded in that session.
7. `POST /api/v1/replays/{event_id}/{origin}/reveal` — allowed only after the forecast was loaded and a valid target artifact has been verified. Records audit metadata.
8. `GET /api/v1/replays/{event_id}/{origin}/actual` — returns target descriptors only after reveal.
9. `GET /api/v1/replays/{event_id}/{origin}/metrics` — serves precomputed database metrics only after reveal.

## Target-separation rules

Before reveal, no response contains the actual-target URI, target raster, next-day observability raster, or target-derived metrics. Direct artifact retrieval is session-bound. Post-reveal artifacts require the same session to have `revealed_at`. Post-reveal files are re-hashed before serving and return `Cache-Control: private, no-store`.

## Static artifact descriptor

```json
{
  "artifact_id": 123,
  "artifact_type": "probability_map",
  "content_type": "image/tiff",
  "url": "/api/v1/artifacts/123",
  "checksum_sha256": "<sha256>",
  "metadata": {
    "format": "GeoTIFF",
    "crs": "EPSG:32610",
    "width": 224,
    "height": 288
  }
}
```

Only allow-listed display/geospatial metadata is returned. Storage paths are never returned.

## Error states

| HTTP | Code | Meaning |
|---:|---|---|
| 403 | `ACTUAL_NOT_REVEALED` | Target or target-derived metrics remain locked |
| 403 | `FORECAST_NOT_VIEWED` | Artifact requested before forecast load |
| 404 | `OOF_ARTIFACT_NOT_PUBLISHED` | Run is OOF but has no published probability map |
| 409 | `REPLAY_NOT_OOF` | Run is not a held-out OOF run |
| 409 | `ARTIFACT_CHECKSUM_MISMATCH` | File bytes differ from registry checksum |
| 409 | `ARTIFACT_SIZE_MISMATCH` | File size differs from registry metadata |
| 410 | `REPLAY_SESSION_EXPIRED` | Session TTL has elapsed |

The replay session gate is intended for the research prototype. A public deployment needs a separate authentication/authorization design.
