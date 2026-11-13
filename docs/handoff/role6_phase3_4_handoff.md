# Role 6 hand-off: Phase 3 and Phase 4 (written during the Prototype integration)

```text
ROLE: Role 6 - Frontend / Product / FIRMS Engineer
PHASE: 3 and 4
STATUS: READY FOR REVIEW (built during integration; no prior Role 6 work existed)

TASK IDS COMPLETED:
- P3-R6-01 Replay lab shell and event catalogue UI (mock-labelled when the backend says mock)
- P3-R6-02 Layer controls: probability, observed fire (locked until reveal), hits/misses;
  uncertainty and explanation shown as "not available for this model"
- P3-R6-03 Replay state flow: event -> day -> session + inputs -> forecast -> reveal -> compare ->
  scores/provenance -> downloads (src/replay/replayState.ts)
- P3-R6-04 Live monitor: FIRMS through the backend only, per-source labels, UTC acquisition time,
  5-day windows, no forecasts on that map
- P3-R6-05 Map/export UX: legends, opacity, nodata, attribution, georeferenced overlays,
  pixel-exact views, downloads only from backend URLs
- P4-R6-01..04 against the real Role 5 API (artifacts are mock until Gate 3)

TASK IDS NOT COMPLETED:
- P1-R6-03 / HLS historical coverage audit with Role 1: NOT RUN
- Citizen reporting and explainability pages: DEFERRED (guide: only after the core replay works;
  explanation layer is UNAVAILABLE per Role 4)

FILES CHANGED:
- frontend/** (new), .github/workflows/frontend-ci.yml (new)

CONFIG / DATA / MODEL VERSIONS:
- API contract: docs/api_contract.md, docs/interfaces/replay_api_contract.md, docs/openapi.json
- Data contract shown in the UI: wsts_strict_input_v1; split hash 34e29090...

COMMANDS RUN:
- npm run typecheck; npm test; npm run build (fresh clone)
- Headless Chromium walk-through of the replay flow against the live demo API

TESTS:
- frontend: 15 passed (state machine, raster/geo helpers, page test recording every request)

RESULTS:
- PASS: reveal disabled until the forecast loads; no /reveal, /actual, /metrics or target
  artifact request before the reveal action; no target URL in the page before reveal
- PASS: mock banner and map watermark on mock replays

ARTIFACTS / EVIDENCE:
- docs/prototype/screenshots/, docs/evidence/integration/prototype_demo_run.json

KNOWN LIMITATIONS:
- Map overlays project the UTM raster corners to lon/lat (display approximation); the pixel
  views and all metrics use the native grid.
- Basemap tiles come from OpenStreetMap and need internet; overlays draw without them.
- Styling is plain CSS rather than Tailwind (plan section 7); nothing depends on it.
- Not tested against live FIRMS (no key in the build environment).

INTEGRATION NOTES:
- Uses only URLs the backend returns; never constructs an artifact URL.
- Real artifacts appear automatically once Role 5 publishes them; no frontend change needed.

FINAL ROLE STATUS:
- CONDITIONAL (Gate 4 needs real published OOF artifacts)
```
