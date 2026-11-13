# FireGuard frontend (Role 6)

React + TypeScript + Vite + MapLibre GL JS, as fixed by the project plan (section on the
engineering stack). Built during the Prototype integration; no Role 6 work existed before it.
Styling is one plain CSS file (`src/styles.css`) instead of Tailwind, to keep the prototype small;
nothing depends on that choice.

## Sections

| Section | Phase 3/4 tasks | What it does |
|---|---|---|
| Overview | P2-R6-01, P4-R6-04 | What FireGuard forecasts, what it does not do, current status |
| Replay lab | P3-R6-01..03/05, P4-R6-01..04 | Choose fire and day, load the out-of-fold forecast, reveal the observed fire, compare, scores, provenance, downloads |
| Live monitor | P2-R6-02, P3-R6-04 | NASA FIRMS detections through the backend proxy, per satellite/product, observation only |
| Method | P2-R6-03, P4-R6-04 | Inputs, models, evaluation protocol, why the target is sealed, data sources |

## Run (PowerShell 5.1, one command per line)

The backend must be running on `http://127.0.0.1:8000` (see `backend/README.md` or
`docs/prototype/RUNBOOK.md`). The dev server proxies `/api` to it.

```powershell
cd frontend
npm install
npm run dev
```

Open http://localhost:5173. Other commands: `npm test` (Vitest), `npm run typecheck`,
`npm run build` (output in `dist/`), `npm run preview`.

Settings go in `frontend/.env.local` (see `.env.example`). Nothing secret belongs there: every
`VITE_` value is compiled into the public bundle. The FIRMS key lives only in `backend/.env`.

## Rules the code keeps (and where they are tested)

- **Forecast first.** The reveal button stays disabled until the forecast has loaded; the replay
  state machine in `src/replay/replayState.ts` ignores a reveal before that
  (`src/__tests__/replayState.test.ts`).
- **No target before reveal.** The page never requests `/reveal`, `/actual`, `/metrics` or a
  target artifact before the reveal action, and no target URL is in the page before it
  (`src/__tests__/ReplayLab.test.tsx` records every request). Artifact URLs are only ever the ones
  the backend returned; the client never builds one.
- **Reset on change.** Picking another fire or day discards the previous session, rasters,
  metrics and reveal state; responses that arrive for an older selection are dropped.
- **Mock is visible.** A replay whose origin is mock (`is_mock`, `registry_status: mock` or
  protocol `MOCK_ONLY`) shows a hatched banner and a map watermark.
- **Unavailable layers are said, not faked.** Uncertainty and explanation layers are listed as
  not available; nothing empty is drawn.
- **Live Monitor is observation only.** Each FIRMS source is separate and labelled, popups show
  satellite, source and UTC acquisition time, requests are capped at 5-day windows, and no
  forecast is ever drawn there. Without a backend FIRMS key the page explains how to add one and
  offers the backend's labelled four-point development fixture.

## How rasters are shown

The backend serves the exact GeoTIFFs Role 3 exported and Role 4 verified. The browser decodes
them with geotiff.js (`src/lib/raster.ts`) and draws them twice:

- on the map, as a MapLibre image source whose corners are the raster's EPSG:32610 corners
  projected with proj4 (`src/lib/geo.ts`). This is a display approximation (a UTM rectangle is
  not exactly a lon/lat quad);
- in the pixel views, one WSTS cell per canvas pixel, which is the exact grid the metrics use.

Probability uses a sequential straw-to-ember ramp, with cells below 1 % left clear. The
comparison view uses the fold's validation threshold (the same one behind F1/IoU) and three
hue-separated classes that do not rely on red versus green.

## Layout

```
src/
  api/types.ts, api/client.ts     typed API client (mirrors backend schemas / docs/openapi.json)
  replay/replayState.ts           replay state machine (pure)
  lib/raster.ts, lib/geo.ts       GeoTIFF decoding, colour maps, georeferencing
  components/MapView.tsx          MapLibre map: raster overlays and FIRMS points
  components/ReplayParts.tsx      pixel canvas, legends, metrics table, provenance, mock banner
  pages/ReplayLab.tsx, LiveMonitor.tsx, InfoPages.tsx
  App.tsx, main.tsx, styles.css
  __tests__/                      Vitest + Testing Library
```
