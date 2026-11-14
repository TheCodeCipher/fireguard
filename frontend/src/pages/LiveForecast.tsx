/**
 * Live forecast: the latest next-day probability maps published by scripts/run_live_forecast.py.
 *
 * Rules kept here:
 *  - Retrieve-only: the page reads /api/v1/live/forecasts and follows only the artifact URLs the
 *    backend returns. Nothing is computed server-side on request; the maps were made offline.
 *  - A forecast, never an observation: this page draws model probabilities only. Satellite
 *    detections stay on the Live monitor tab, which is observation-only by contract.
 *  - Honest labelling: a fixed research disclaimer, a red banner when Gate 2 was not passed
 *    (UNGATED) or the run is a mock, and an amber one when the run is stale or a region's model
 *    inputs were mostly missing (the model then saw training-mean values there).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, api, fetchLiveArtifactBytes } from "../api/client";
import type { LiveForecast, LiveRegion } from "../api/types";
import { MapView, type RasterOverlay } from "../components/MapView";
import { ProbabilityLegend } from "../components/ReplayParts";
import { rasterCorners } from "../lib/geo";
import { decodeGeoTiff, probabilityColor, rgbaToDataUrl, toImageData } from "../lib/raster";

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

export function LiveForecastPage() {
  const [run, setRun] = useState<LiveForecast | null>(null);
  const [error, setError] = useState<{ code: string; message: string } | null>(null);
  const [busy, setBusy] = useState(true);
  const [layer, setLayer] = useState<Layer>("probability");
  const [focus, setFocus] = useState<string | null>(null);
  const [images, setImages] = useState<Record<string, string>>({});

  const load = useCallback(async () => {
    setBusy(true);
    setError(null);
    try {
      setRun(await api.liveForecast());
    } catch (e) {
      setRun(null);
      setError(e instanceof ApiError ? { code: e.code, message: e.message } : { code: "CLIENT", message: String(e) });
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
          if (!cancelled)
            setError(
              e instanceof ApiError ? { code: e.code, message: e.message } : { code: "CLIENT", message: String(e) },
            );
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

  const fit = useMemo(() => {
    if (!run) return null;
    const r = run.regions.find((x) => x.region_id === focus);
    return r ? r.bbox_wgs84 : unionBounds(run.regions);
  }, [run, focus]);

  const notPublished = error?.code === "LIVE_FORECAST_NOT_AVAILABLE";

  return (
    <div className="monitor">
      <aside className="monitor-controls" aria-label="Live forecast">
        <div>
          <h1 className="rail-title">Live forecast</h1>
          <p className="domain-note" style={{ marginBottom: 0 }}>
            Next-day probability that VIIRS detects fire in each 375 m cell, for today's active fires. Made
            offline once a day from satellite inputs; this page only displays it.
          </p>
        </div>

        {busy ? <p className="fine">Loading the latest forecast…</p> : null}
        {notPublished ? (
          <p className="fine" role="status">
            No live forecast yet. It appears here after the daily forecast job has run.
          </p>
        ) : error ? (
          <div className="error" role="alert">
            <p>
              {error.message} ({error.code})
            </p>
            <button type="button" onClick={load}>
              Retry
            </button>
          </div>
        ) : null}

        {run ? (
          <>
            <p className="fine">
              Forecast for <strong>{run.target_date}</strong> (UTC), from inputs up to {run.origin_date}. Model{" "}
              {run.model_version}, average of {run.folds.length} fold models. Made{" "}
              {new Date(run.generated_at).toLocaleString()}.
            </p>
            <label>
              Map layer
              <select value={layer} onChange={(e) => setLayer(e.target.value as Layer)}>
                <option value="probability">Forecast probability</option>
                <option value="fold_spread">Disagreement between fold models</option>
              </select>
            </label>
            {layer === "probability" ? (
              <ProbabilityLegend />
            ) : (
              <p className="fine">
                Grey shows where the fold models disagree (largest minus smallest probability). It is not an
                uncertainty estimate.
              </p>
            )}
            <h2 className="rail-title" style={{ fontSize: "1rem" }}>
              Fire regions ({run.regions.length})
            </h2>
            {run.regions.length === 0 ? <p className="fine">No active fire cluster inside the model's area today.</p> : null}
            <ul className="region-list" aria-label="Fire regions" style={{ listStyle: "none", padding: 0, margin: 0 }}>
              {run.regions.map((r) => {
                const miss = missingReflectance(r);
                return (
                  <li key={r.region_id} style={{ marginBottom: "0.6rem" }}>
                    <button
                      type="button"
                      aria-pressed={focus === r.region_id}
                      onClick={() => setFocus(focus === r.region_id ? null : r.region_id)}
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
          </>
        ) : null}
      </aside>

      <section className="stage">
        {run && (run.gate_status !== "GATED" || run.is_mock) ? (
          <div className="mock-banner" role="note" data-testid="ungated-banner">
            <strong>{run.is_mock ? "Mock run." : "Not validated."}</strong>{" "}
            {run.is_mock
              ? "Synthetic test checkpoints. This map shows the pipeline working, not a forecast."
              : "The model has not passed Gate 2, so this map is for development only."}
          </div>
        ) : null}
        {run && (run.stale || run.regions.some((r) => missingReflectance(r) > NAN_WARN)) ? (
          <div className="mock-banner" role="note" data-testid="caution-banner" style={{ background: "#fdf3d8" }}>
            <strong>Use with care.</strong>{" "}
            {run.stale ? `This forecast is more than ${run.stale_after_hours} hours old. ` : ""}
            {run.regions.some((r) => missingReflectance(r) > NAN_WARN)
              ? "Some regions had little surface-reflectance data; the model used training averages there."
              : ""}
          </div>
        ) : null}
        <MapView ariaLabel="Map of the live next-day fire forecast" overlays={overlays} fitBounds={fit} />
        <p className="attribution fine">{run ? run.disclaimer : "Research forecast, not an official warning."}</p>
      </section>
    </div>
  );
}
