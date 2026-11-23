/**
 * Live: today's fires and FireGuard's forecast for tomorrow, on one map.
 *
 * Two layers, each switchable and labelled separately so they are never confused:
 *  - Forecast (FireGuard): the next-day probability maps published once a day by
 *    scripts/live/run_live_forecast.py with the best trained model (results/best_model.json). The page
 *    only reads /api/v1/live/forecasts and the artifact URLs the backend returns; nothing is
 *    computed on request.
 *  - Detections (NASA FIRMS): satellite observations through the backend proxy
 *    (/api/v1/fires); the FIRMS key stays on the server. One satellite/product at a time, never
 *    merged, never presented as a forecast. Requests are limited to 5-day windows.
 *
 * Honest labelling: a fixed research disclaimer, a red banner when the model did not match or
 * beat the strongest baseline (UNGATED) or the run is a mock, an amber one when the run is stale
 * or a region's inputs were mostly missing.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, api, fetchLiveArtifactBytes } from "../api/client";
import { FIRMS_SOURCES, type FireObservation, type FirmsSource, type LiveForecast, type LiveRegion } from "../api/types";
import { MapView, type PointFeature, type RasterOverlay } from "../components/MapView";
import { ProbabilityLegend } from "../components/ReplayParts";
import { rasterCorners } from "../lib/geo";
import { decodeGeoTiff, probabilityColor, rgbaToDataUrl, toImageData } from "../lib/raster";

type Err = { code: string; message: string };
const toErr = (e: unknown): Err =>
  e instanceof ApiError ? { code: e.code, message: e.message } : { code: "CLIENT", message: String(e) };

// ------------------------------------------------------------------------------------ forecast

const NAN_WARN = 0.5;
const REFLECTANCE = ["viirs_m11", "viirs_i2", "viirs_i1"];
type Layer = "probability" | "fold_spread";

/** Fold spread (max - min over the fold models) in a neutral grey ramp, transparent near 0. */
function spreadColor(v: number): [number, number, number, number] {
  if (!Number.isFinite(v) || v < 0.02) return [0, 0, 0, 0];
  const x = Math.min(1, v / 0.5);
  const g = Math.round(150 - 120 * x);
  return [g, g, g, Math.round(90 + 150 * x)];
}

function missingReflectance(r: LiveRegion): number {
  const vals = REFLECTANCE.map((c) => r.nan_fraction[c]).filter((v) => typeof v === "number");
  return vals.length ? Math.max(...vals) : 0;
}

function unionBounds(regions: LiveRegion[]): [number, number, number, number] | null {
  if (!regions.length) return null;
  return [
    Math.min(...regions.map((r) => r.bbox_wgs84[0])),
    Math.min(...regions.map((r) => r.bbox_wgs84[1])),
    Math.max(...regions.map((r) => r.bbox_wgs84[2])),
    Math.max(...regions.map((r) => r.bbox_wgs84[3])),
  ];
}

function useLiveForecast() {
  const [run, setRun] = useState<LiveForecast | null>(null);
  const [error, setError] = useState<Err | null>(null);
  const [busy, setBusy] = useState(true);
  const [layer, setLayer] = useState<Layer>("probability");
  const [images, setImages] = useState<Record<string, string>>({});

  const load = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      setRun(await api.liveForecast());
    } catch (e) {
      setRun(null);
      setError(toErr(e));
    } finally {
      setBusy(false);
    }
  }, []);
  useEffect(() => {
    void load();
  }, [load]);

  // Decode the chosen layer of every region (one GeoTIFF per region, from backend URLs only).
  useEffect(() => {
    if (!run) return;
    let cancelled = false;
    setImages({});
    void (async () => {
      const out: Record<string, string> = {};
      for (const r of run.regions) {
        try {
          const raster = await decodeGeoTiff(await fetchLiveArtifactBytes(r.artifacts[layer].url));
          const color = layer === "probability" ? probabilityColor : spreadColor;
          const rgba = toImageData(raster.width, raster.height, (i) => color(raster.data[i]));
          out[r.region_id] = rgbaToDataUrl(raster.width, raster.height, rgba);
        } catch (e) {
          if (!cancelled) setError(toErr(e));
        }
      }
      if (!cancelled) setImages(out);
    })();
    return () => {
      cancelled = true;
    };
  }, [run, layer]);

  const overlays: RasterOverlay[] = useMemo(() => {
    if (!run) return [];
    return run.regions.flatMap((r) => {
      const corners = rasterCorners(r.crs, r.transform, r.width, r.height);
      const dataUrl = images[r.region_id];
      return corners && dataUrl ? [{ id: `live-${r.region_id}`, dataUrl, corners, opacity: 0.85, visible: true }] : [];
    });
  }, [run, images]);

  return { run, error, busy, layer, setLayer, overlays, load };
}

// ------------------------------------------------------------------------------------ detections

const SOURCE_LABEL: Record<FirmsSource, string> = {
  VIIRS_SNPP_NRT: "Suomi-NPP VIIRS, near real time",
  VIIRS_SNPP_SP: "Suomi-NPP VIIRS, standard processing",
  VIIRS_NOAA20_NRT: "NOAA-20 VIIRS, near real time",
  VIIRS_NOAA20_SP: "NOAA-20 VIIRS, standard processing",
  VIIRS_NOAA21_NRT: "NOAA-21 VIIRS, near real time",
  VIIRS_NOAA21_SP: "NOAA-21 VIIRS, standard processing",
};

const AREAS: Record<string, { label: string; bbox: string }> = {
  west: { label: "Western United States", bbox: "-125,31,-102,49" },
  ca: { label: "California", bbox: "-124.5,32.5,-114,42" },
  pnw: { label: "Pacific Northwest", bbox: "-125,42,-116,49" },
  world: { label: "Whole world", bbox: "-180,-90,180,90" },
};

const MAX_DAYS = 5;
const isoDay = (d: Date) => d.toISOString().slice(0, 10);
const daysBetween = (a: string, b: string) => Math.round((Date.parse(b) - Date.parse(a)) / 86_400_000);

function formatAge(seconds: number | null): string {
  if (seconds === null) return "unknown";
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  if (seconds < 86_400) return `${(seconds / 3600).toFixed(1)} h`;
  return `${(seconds / 86_400).toFixed(1)} days`;
}

const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

/** Popup: source, platform and acquisition time are always shown. */
function popup(p: Record<string, unknown>): string {
  const row = (k: string, v: unknown) =>
    `<tr><th>${k}</th><td>${escapeHtml(v === null || v === undefined || v === "" ? "–" : String(v))}</td></tr>`;
  return `<table class="popup">${row("Satellite", p.platform)}${row("Source", p.source)}${row(
    "Acquired (UTC)",
    `${p.acquisition_date} ${String(p.acquisition_time).replace(/(\d\d)(\d\d)/, "$1:$2")}`,
  )}${row("Confidence", p.confidence)}${row("FRP (MW)", p.frp)}${row("Day/night", p.daynight)}</table>
  <p class="popup-note">Satellite observation, not a FireGuard forecast.</p>`;
}

function useDetections() {
  const today = new Date();
  const [source, setSource] = useState<FirmsSource>("VIIRS_SNPP_NRT");
  const [end, setEnd] = useState(isoDay(today));
  const [start, setStart] = useState(isoDay(new Date(today.getTime() - 86_400_000)));
  const [area, setArea] = useState("west");
  const [confidence, setConfidence] = useState("");
  const [obs, setObs] = useState<FireObservation[] | null>(null);
  const [meta, setMeta] = useState<string | null>(null);
  const [fixture, setFixture] = useState(false);
  const [error, setError] = useState<Err | null>(null);
  const [busy, setBusy] = useState(false);
  const [latest, setLatest] = useState<string | null>(null);

  // Default the dates to the newest day FIRMS has for this source ("today" can be past its feed).
  useEffect(() => {
    let live = true;
    api
      .firmsAvailability(source)
      .then((a) => {
        if (!live) return;
        setLatest(a.max_date);
        setEnd(a.max_date);
        const prev = isoDay(new Date(Date.parse(a.max_date) - 86_400_000));
        setStart(prev < a.min_date ? a.min_date : prev);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [source]);

  const span = daysBetween(start, end);
  const rangeProblem =
    span < 0 ? "The end date is before the start date." : span >= MAX_DAYS ? `Choose at most ${MAX_DAYS} days.` : null;

  const load = useCallback(
    async (range?: { start: string; end: string; bbox?: string }) => {
      setBusy(true);
      setError(null);
      setFixture(false);
      try {
        const r = await api.fires({
          source,
          start: range?.start ?? start,
          end: range?.end ?? end,
          bbox: range?.bbox ?? AREAS[area].bbox,
          confidence: confidence || undefined,
        });
        setObs(r.observations);
        setMeta(
          `${r.total_count.toLocaleString()} detections from ${r.platform} (${r.product}), ${r.start_date} to ${r.end_date}. ` +
            `Newest is ${formatAge(r.data_age_seconds)} old. Retrieved ${new Date(r.retrieved_at).toLocaleString()}` +
            (r.has_more ? `; showing the first ${r.count.toLocaleString()}.` : "."),
        );
      } catch (e) {
        setObs(null);
        setMeta(null);
        setError(toErr(e));
      } finally {
        setBusy(false);
      }
    },
    [source, start, end, area, confidence],
  );

  /** The backend's four-point development fixture, for working without a FIRMS key. */
  const loadFixture = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      const r = await api.mockFires();
      setObs(r.observations);
      setFixture(true);
      setMeta(`Development fixture: ${r.count} made-up points (${r.source}).`);
    } catch (e) {
      setError(toErr(e));
    } finally {
      setBusy(false);
    }
  }, []);

  const points: PointFeature[] | undefined = useMemo(
    () =>
      obs?.map((o) => ({
        lon: o.longitude,
        lat: o.latitude,
        properties: {
          platform: o.platform,
          source: o.source,
          acquisition_date: o.acquisition_date,
          acquisition_time: o.acquisition_time,
          confidence: o.confidence,
          frp: o.frp,
          daynight: o.daynight,
        },
      })),
    [obs],
  );

  return {
    source, setSource, start, setStart, end, setEnd, area, setArea, confidence, setConfidence,
    latest, today, rangeProblem, load, loadFixture, points, meta, fixture, error, busy,
  };
}

// ------------------------------------------------------------------------------------ page

export function Live() {
  const fc = useLiveForecast();
  const det = useDetections();
  const [showForecast, setShowForecast] = useState(true);
  const [showDetections, setShowDetections] = useState(true);
  const [focus, setFocus] = useState<string | null>(null);
  const [fitTo, setFitTo] = useState<"forecast" | "area">("forecast");
  const run = fc.run;

  const fit = useMemo(() => {
    if (fitTo === "forecast" && run && run.regions.length) {
      const r = run.regions.find((x) => x.region_id === focus);
      return r ? r.bbox_wgs84 : unionBounds(run.regions);
    }
    return det.area === "world" ? null : (AREAS[det.area].bbox.split(",").map(Number) as [number, number, number, number]);
  }, [fitTo, run, focus, det.area]);

  const notPublished = fc.error?.code === "LIVE_FORECAST_NOT_AVAILABLE";
  const lowInputs = run ? run.regions.some((r) => missingReflectance(r) > NAN_WARN) : false;

  return (
    <div className="monitor">
      <aside className="monitor-controls" aria-label="Live forecast and detections">
        <div>
          <h1 className="rail-title">Live</h1>
          <p className="domain-note" style={{ marginBottom: 0 }}>
            FireGuard's forecast of where VIIRS will detect fire tomorrow, next to the satellite detections. The two
            layers are separate: one is a model forecast, the other is an observation.
          </p>
        </div>

        <section aria-label="Forecast">
          <h2 className="rail-title" style={{ fontSize: "1rem" }}>
            Forecast (FireGuard)
          </h2>
          <label className="check">
            <input type="checkbox" checked={showForecast} onChange={(e) => setShowForecast(e.target.checked)} /> Show
            forecast layer
          </label>
          {fc.busy ? <p className="fine">Loading the latest forecast…</p> : null}
          {notPublished ? (
            <p className="fine" role="status">
              No live forecast yet. It appears here after the daily forecast job has run.
            </p>
          ) : fc.error ? (
            <div className="error" role="alert">
              <p>
                {fc.error.message} ({fc.error.code})
              </p>
              <button type="button" onClick={fc.load}>
                Retry
              </button>
            </div>
          ) : null}
          {run ? (
            <>
              <p className="fine">
                Forecast for <strong>{run.target_date}</strong> (UTC), from inputs up to {run.origin_date}. Model{" "}
                {run.model_version} ({run.architecture}), average of {run.folds.length} fold models. Made{" "}
                {new Date(run.generated_at).toLocaleString()}.
              </p>
              <label>
                Map layer
                <select value={fc.layer} onChange={(e) => fc.setLayer(e.target.value as Layer)}>
                  <option value="probability">Forecast probability</option>
                  <option value="fold_spread">Disagreement between fold models</option>
                </select>
              </label>
              {fc.layer === "probability" ? (
                <ProbabilityLegend />
              ) : (
                <p className="fine">
                  Grey shows where the fold models disagree (largest minus smallest probability). It is not an
                  uncertainty estimate.
                </p>
              )}
              <p className="fine">Fire regions ({run.regions.length})</p>
              {run.regions.length === 0 ? (
                <p className="fine">No active fire cluster inside the model's area today.</p>
              ) : null}
              <ul className="region-list" aria-label="Fire regions" style={{ listStyle: "none", padding: 0, margin: 0 }}>
                {run.regions.map((r) => {
                  const miss = missingReflectance(r);
                  return (
                    <li key={r.region_id} style={{ marginBottom: "0.6rem" }}>
                      <button
                        type="button"
                        aria-pressed={focus === r.region_id}
                        onClick={() => {
                          setFitTo("forecast");
                          setFocus(focus === r.region_id ? null : r.region_id);
                        }}
                      >
                        {r.center_lat.toFixed(2)}°N {Math.abs(r.center_lon).toFixed(2)}°
                        {r.center_lon < 0 ? "W" : "E"}
                      </button>{" "}
                      <span className="fine">
                        {r.n_detections} detections
                        {miss > NAN_WARN ? ` · surface reflectance ${Math.round(miss * 100)}% missing` : ""}
                      </span>
                    </li>
                  );
                })}
              </ul>
              <button
                type="button"
                onClick={() => {
                  setShowDetections(true);
                  setFitTo("forecast");
                  const b = unionBounds(run.regions);
                  void det.load({ start: run.target_date, end: run.target_date, bbox: b ? b.join(",") : undefined });
                }}
                disabled={det.busy}
              >
                Check against detections on {run.target_date}
              </button>
            </>
          ) : null}
        </section>

        <section aria-label="Satellite detections">
          <h2 className="rail-title" style={{ fontSize: "1rem" }}>
            Detections (NASA FIRMS)
          </h2>
          <label className="check">
            <input type="checkbox" checked={showDetections} onChange={(e) => setShowDetections(e.target.checked)} />{" "}
            Show detection layer
          </label>
          <label>
            Satellite and product
            <select value={det.source} onChange={(e) => det.setSource(e.target.value as FirmsSource)}>
              {FIRMS_SOURCES.map((s) => (
                <option key={s} value={s}>
                  {SOURCE_LABEL[s]}
                </option>
              ))}
            </select>
          </label>
          <label>
            Area
            <select value={det.area} onChange={(e) => det.setArea(e.target.value)}>
              {Object.entries(AREAS).map(([k, r]) => (
                <option key={k} value={k}>
                  {r.label}
                </option>
              ))}
            </select>
          </label>
          <div className="date-pair">
            <label>
              From (UTC)
              <input type="date" value={det.start} max={det.end} onChange={(e) => det.setStart(e.target.value)} />
            </label>
            <label>
              To (UTC)
              <input
                type="date"
                value={det.end}
                max={det.latest ?? isoDay(det.today)}
                onChange={(e) => det.setEnd(e.target.value)}
              />
            </label>
          </div>
          <label>
            Detection confidence
            <select value={det.confidence} onChange={(e) => det.setConfidence(e.target.value)}>
              <option value="">Any</option>
              <option value="high">High</option>
              <option value="nominal">Nominal</option>
              <option value="low">Low</option>
            </select>
          </label>
          {det.rangeProblem ? <p className="error">{det.rangeProblem}</p> : null}
          <button
            type="button"
            className="primary"
            onClick={() => {
              setShowDetections(true);
              setFitTo("area");
              void det.load();
            }}
            disabled={det.busy || !!det.rangeProblem}
          >
            Show detections
          </button>
          {det.error ? (
            <div className="error" role="alert">
              {det.error.code === "FIRMS_NOT_CONFIGURED" ? (
                <>
                  <p>
                    The backend has no FIRMS key yet. Put <code>FIRMS_MAP_KEY=…</code> in <code>backend/.env</code> and
                    restart the API.
                  </p>
                  <button type="button" onClick={det.loadFixture}>
                    Show the development fixture instead
                  </button>
                </>
              ) : (
                <p>
                  {det.error.message} ({det.error.code})
                </p>
              )}
            </div>
          ) : null}
          {det.meta ? <p className="fine">{det.meta}</p> : null}
          <ul className="confidence-legend" aria-label="Point colours">
            <li><span className="dot high" /> High confidence</li>
            <li><span className="dot nominal" /> Nominal</li>
            <li><span className="dot low" /> Low</li>
            <li><span className="dot unknown" /> Not reported</li>
          </ul>
        </section>
      </aside>

      <section className="stage">
        {run && (run.gate_status !== "GATED" || run.is_mock) ? (
          <div className="mock-banner" role="note" data-testid="ungated-banner">
            <strong>{run.is_mock ? "Mock run." : "Not validated."}</strong>{" "}
            {run.is_mock
              ? "Synthetic test checkpoints. This map shows the pipeline working, not a forecast."
              : "The model did not match or beat the strongest simple baseline, so this map is for development only."}
          </div>
        ) : null}
        {run && (run.stale || lowInputs) ? (
          <div className="mock-banner" role="note" data-testid="caution-banner" style={{ background: "#fdf3d8" }}>
            <strong>Use with care.</strong>{" "}
            {run.stale ? `This forecast is more than ${run.stale_after_hours} hours old. ` : ""}
            {lowInputs ? "Some regions had little surface-reflectance data; the model used training averages there." : ""}
          </div>
        ) : null}
        {det.fixture ? (
          <div className="mock-banner" role="note">
            <strong>Development fixture.</strong> Four made-up points from the backend, not NASA observations.
          </div>
        ) : null}
        <MapView
          ariaLabel="Map of the next-day fire forecast and satellite fire detections"
          overlays={showForecast ? fc.overlays : []}
          points={showDetections ? det.points : undefined}
          fitBounds={fit}
          pointPopup={popup}
          watermark={det.fixture ? "Development fixture" : null}
        />
        <p className="attribution fine">
          {run ? run.disclaimer : "Research forecast, not an official warning."} Fire detections: NASA FIRMS (LANCE),
          UTC acquisition times; near-real-time products are replaced by standard processing later.
        </p>
      </section>
    </div>
  );
}
