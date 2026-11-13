/**
 * Forecast & Replay Lab (Role 6, P3-R6-01..05 and P4-R6-01..04).
 *
 * Replays a held-out WSTS event day: the user loads the precomputed out-of-fold forecast first,
 * and only then may reveal what the satellite actually detected the next day. All data comes
 * from Role 5's retrieve-only API; nothing is predicted in the browser or on the server.
 *
 * Leakage rules this page keeps (and the tests in src/__tests__ check):
 *  - no request for the actual target, its artifact or the metrics before the reveal action;
 *  - the reveal button is disabled until the forecast has loaded (replayState.canReveal);
 *  - changing event or day discards every raster, metric and session of the previous replay;
 *  - uncertainty and explanation layers are shown as unavailable, never as empty overlays.
 */
import { useCallback, useEffect, useMemo, useReducer, useRef, useState } from "react";
import { ApiError, api, fetchArtifactBytes } from "../api/client";
import type { ArtifactDescriptor, ReplayOrigin, WstsEvent } from "../api/types";
import { MapView, type RasterOverlay } from "../components/MapView";
import {
  ComparisonLegend,
  MetricsTable,
  MockBanner,
  ProbabilityLegend,
  ProvenanceList,
  RasterCanvas,
} from "../components/ReplayParts";
import { cornersBounds, rasterCorners } from "../lib/geo";
import {
  comparisonColor,
  comparisonCounts,
  decodeGeoTiff,
  type DecodedRaster,
  probabilityColor,
  rgbaToDataUrl,
  targetColor,
  toImageData,
} from "../lib/raster";
import {
  canLoadForecast,
  canReveal,
  initialReplayState,
  isMockReplay,
  replayReducer,
} from "../replay/replayState";

/** Decoded raster plus the artifact descriptor it came from (for georeferencing/downloads). */
interface LoadedRaster {
  artifact: ArtifactDescriptor;
  raster: DecodedRaster;
}

function errorText(e: unknown): { code: string; message: string } {
  if (e instanceof ApiError) return { code: e.code, message: e.message };
  return { code: "CLIENT", message: e instanceof Error ? e.message : "Unexpected error" };
}

/** Save bytes the backend served as a file (downloads respect reveal state: the backend refuses
 *  the target before reveal, so the button for it is not even rendered until then). */
function saveBlob(bytes: ArrayBuffer | string, filename: string, type: string) {
  const url = URL.createObjectURL(new Blob([bytes], { type }));
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function ReplayLab() {
  const [state, dispatch] = useReducer(replayReducer, initialReplayState);
  const [events, setEvents] = useState<WstsEvent[] | null>(null);
  const [eventsError, setEventsError] = useState<string | null>(null);
  const [origins, setOrigins] = useState<ReplayOrigin[] | null>(null);
  const [prob, setProb] = useState<LoadedRaster | null>(null);
  const [actual, setActual] = useState<LoadedRaster | null>(null);
  const [show, setShow] = useState({ probability: true, actual: true, comparison: false });
  const [opacity, setOpacity] = useState(0.85);
  // Identifies the replay a response belongs to; responses for an older selection are dropped.
  const generation = useRef(0);

  // Catalogue: only events with a replayable OOF forecast are offered (P4-R6-01).
  useEffect(() => {
    api
      .events()
      .then((r) => setEvents(r.events.filter((e) => e.has_replay)))
      .catch((e) => setEventsError(errorText(e).message));
  }, []);

  const chooseEvent = useCallback((eventId: string) => {
    generation.current++;
    dispatch({ type: "selectEvent", eventId });
    setOrigins(null);
    setProb(null);
    setActual(null);
    const gen = generation.current;
    api
      .origins(eventId)
      .then((r) => gen === generation.current && setOrigins(r.origins))
      .catch((e) => gen === generation.current && dispatch({ type: "failed", ...errorText(e) }));
  }, []);

  const chooseOrigin = useCallback(
    async (origin: ReplayOrigin) => {
      if (!state.eventId) return;
      generation.current++;
      const gen = generation.current;
      dispatch({ type: "selectOrigin", origin });
      setProb(null);
      setActual(null);
      setShow({ probability: true, actual: true, comparison: false });
      dispatch({ type: "busy" });
      try {
        // A new session per replay; the backend ties forecast-viewed and revealed to it.
        const session = await api.startSession(state.eventId, origin.origin_date);
        const inputs = await api.inputs(state.eventId, origin.origin_date, session.session_id);
        if (gen === generation.current) dispatch({ type: "sessionReady", session, inputs });
      } catch (e) {
        if (gen === generation.current) dispatch({ type: "failed", ...errorText(e) });
      }
    },
    [state.eventId],
  );

  const loadForecast = useCallback(async () => {
    const { eventId, origin, session } = state;
    if (!eventId || !origin || !session) return;
    const gen = generation.current;
    dispatch({ type: "busy" });
    try {
      const forecast = await api.forecast(eventId, origin.origin_date, session.session_id);
      const artifact = forecast.artifacts.find((a) => a.artifact_type === "probability_map");
      if (!artifact) throw new ApiError(404, "NO_PROBABILITY", "This forecast has no probability map.");
      const bytes = await fetchArtifactBytes(artifact.url, session.session_id);
      const raster = await decodeGeoTiff(bytes, artifact.metadata.nodata);
      if (gen !== generation.current) return;
      setProb({ artifact, raster });
      dispatch({ type: "forecastLoaded", forecast });
    } catch (e) {
      if (gen === generation.current) dispatch({ type: "failed", ...errorText(e) });
    }
  }, [state]);

  const reveal = useCallback(async () => {
    const { eventId, origin, session } = state;
    if (!eventId || !origin || !session || !canReveal(state)) return;
    const gen = generation.current;
    dispatch({ type: "busy" });
    try {
      await api.reveal(eventId, origin.origin_date, session.session_id);
      const actualResp = await api.actual(eventId, origin.origin_date, session.session_id);
      const metrics = await api.metrics(eventId, origin.origin_date, session.session_id);
      const artifact = actualResp.artifacts.find((a) => a.artifact_type === "actual_target");
      if (artifact) {
        const bytes = await fetchArtifactBytes(artifact.url, session.session_id);
        const raster = await decodeGeoTiff(bytes, artifact.metadata.nodata ?? 255);
        if (gen !== generation.current) return;
        setActual({ artifact, raster });
      }
      if (gen !== generation.current) return;
      setShow({ probability: true, actual: true, comparison: false });
      dispatch({ type: "revealed", actual: actualResp, metrics: metrics.metrics });
    } catch (e) {
      if (gen === generation.current) dispatch({ type: "failed", ...errorText(e) });
    }
  }, [state]);

  // The fold's validation threshold arrives with the (post-reveal) metrics.
  const threshold = state.metrics?.find((m) => m.threshold !== null)?.threshold ?? null;

  // Paint the decoded rasters once per change; reused by the map and the pixel views.
  const painted = useMemo(() => {
    if (!prob) return null;
    const { width, height, data } = prob.raster;
    const probRgba = toImageData(width, height, (i) => probabilityColor(data[i]));
    const actRgba = actual ? toImageData(width, height, (i) => targetColor(actual.raster.data[i])) : null;
    const cmpRgba =
      actual && threshold !== null
        ? toImageData(width, height, (i) => comparisonColor(data[i], actual.raster.data[i], threshold))
        : null;
    const counts = actual && threshold !== null ? comparisonCounts(prob.raster, actual.raster, threshold) : undefined;
    return { width, height, probRgba, actRgba, cmpRgba, counts };
  }, [prob, actual, threshold]);

  const corners = useMemo(() => {
    if (!prob) return null;
    const m = prob.artifact.metadata;
    return rasterCorners(m.crs, m.affine_transform, prob.raster.width, prob.raster.height);
  }, [prob]);

  const overlays: RasterOverlay[] = useMemo(() => {
    if (!painted || !corners) return [];
    const list: RasterOverlay[] = [
      {
        id: "probability",
        dataUrl: rgbaToDataUrl(painted.width, painted.height, painted.probRgba),
        corners,
        opacity,
        visible: show.probability,
      },
    ];
    // The target overlay only exists after reveal; before that there is nothing to pass in.
    if (state.stage === "revealed" && painted.actRgba) {
      list.push({
        id: "actual",
        dataUrl: rgbaToDataUrl(painted.width, painted.height, painted.actRgba),
        corners,
        opacity: 1,
        visible: show.actual,
      });
    }
    if (state.stage === "revealed" && painted.cmpRgba) {
      list.push({
        id: "comparison",
        dataUrl: rgbaToDataUrl(painted.width, painted.height, painted.cmpRgba),
        corners,
        opacity: 1,
        visible: show.comparison,
      });
    }
    return list;
  }, [painted, corners, opacity, show, state.stage]);

  const mock = isMockReplay(state);
  const { origin, forecast, session } = state;

  return (
    <div className="replay">
      <aside className="steps" aria-label="Replay steps">
        <p className="domain-note">
          Historical replay of western-US wildfires from the WildfireSpreadTS benchmark (2018–2021). Each
          forecast was made by a model that never saw this fire during training.
        </p>

        <ol className="step-list">
          <li className={state.stage === "chooseEvent" ? "current" : "done"}>
            <h3>Choose a fire</h3>
            {eventsError ? <p className="error">{eventsError}</p> : null}
            {events === null && !eventsError ? <p className="fine">Loading the event catalogue…</p> : null}
            {events && events.length === 0 ? (
              <p className="fine">
                No replays are registered yet. Run the offline pipeline and load its registry bundle
                (docs/prototype/RUNBOOK.md).
              </p>
            ) : null}
            {events && events.length > 0 ? (
              <select
                aria-label="Fire event"
                value={state.eventId ?? ""}
                onChange={(e) => e.target.value && chooseEvent(e.target.value)}
              >
                <option value="">Select an event…</option>
                {events.map((e) => (
                  <option key={e.event_id} value={e.event_id}>
                    {e.event_id}, {e.year}
                  </option>
                ))}
              </select>
            ) : null}
          </li>

          <li className={!state.eventId ? "todo" : state.stage === "chooseDay" ? "current" : "done"}>
            <h3>Choose the forecast day</h3>
            {state.eventId && origins === null ? <p className="fine">Loading available days…</p> : null}
            {origins && origins.length > 0 ? (
              <div className="day-grid" role="group" aria-label="Forecast days">
                {origins.map((o) => (
                  <button
                    key={o.origin_date}
                    type="button"
                    className={origin?.origin_date === o.origin_date ? "day selected" : "day"}
                    onClick={() => chooseOrigin(o)}
                    title={`Inputs up to ${o.origin_date}; forecast for ${o.target_date}`}
                  >
                    {o.target_date.slice(5)}
                  </button>
                ))}
              </div>
            ) : null}
            {origins && origins.length > 0 ? (
              <p className="fine">Each button is the day being forecast, using the five days before it.</p>
            ) : null}
          </li>

          <li className={state.stage === "ready" ? "current" : state.stage === "forecast" || state.stage === "revealed" ? "done" : "todo"}>
            <h3>Load the forecast</h3>
            {origin && state.inputs ? (
              <p className="fine">
                Inputs: {String(state.inputs.input_metadata.channel_manifest ?? "strict channel list")}, from the
                five days up to {origin.origin_date}.
              </p>
            ) : null}
            <button type="button" className="primary" disabled={!canLoadForecast(state)} onClick={loadForecast}>
              Load forecast
            </button>
          </li>

          <li className={state.stage === "forecast" ? "current" : state.stage === "revealed" ? "done" : "todo"}>
            <h3>Reveal what happened</h3>
            <p className="fine">Unlocks the satellite detections of the forecast day and the scores.</p>
            <button type="button" className="primary" disabled={!canReveal(state)} onClick={reveal}>
              Reveal actual fire
            </button>
          </li>
        </ol>

        {state.busy ? <p className="fine" aria-live="polite">Working…</p> : null}
        {state.error ? (
          <p className="error" role="alert">
            {state.error.code === "REPLAY_SESSION_EXPIRED"
              ? "This replay session expired. Choose the day again to start a new one."
              : `${state.error.message} (${state.error.code})`}
          </p>
        ) : null}
      </aside>

      <section className="stage" aria-label="Replay view">
        {mock ? <MockBanner /> : null}

        <MapView
          ariaLabel="Map of the forecast and, after reveal, the observed fire"
          overlays={overlays}
          fitBounds={corners ? cornersBounds(corners) : null}
          watermark={mock ? "Synthetic demonstration" : null}
        />

        {prob ? (
          <div className="layer-controls" role="group" aria-label="Map layers">
            <label>
              <input
                type="checkbox"
                checked={show.probability}
                onChange={(e) => setShow({ ...show, probability: e.target.checked })}
              />
              Forecast probability
            </label>
            <label className="opacity">
              Opacity
              <input
                type="range"
                min={0.2}
                max={1}
                step={0.05}
                value={opacity}
                onChange={(e) => setOpacity(Number(e.target.value))}
              />
            </label>
            <label className={state.stage === "revealed" ? "" : "locked"}>
              <input
                type="checkbox"
                disabled={state.stage !== "revealed"}
                checked={state.stage === "revealed" && show.actual}
                onChange={(e) => setShow({ ...show, actual: e.target.checked })}
              />
              Observed fire {state.stage === "revealed" ? "" : "(sealed until reveal)"}
            </label>
            <label className={state.stage === "revealed" ? "" : "locked"}>
              <input
                type="checkbox"
                disabled={state.stage !== "revealed"}
                checked={state.stage === "revealed" && show.comparison}
                onChange={(e) => setShow({ ...show, comparison: e.target.checked })}
              />
              Hits and misses
            </label>
            <span className="unavailable" title="Not implemented or not validated for this model">
              Uncertainty and explanation layers: not available for this model
            </span>
          </div>
        ) : null}

        {painted ? (
          <div className="pixel-views">
            <figure>
              <RasterCanvas
                width={painted.width}
                height={painted.height}
                rgba={painted.probRgba}
                label="Forecast probability, native grid"
              />
              <figcaption>
                Forecast for {origin?.target_date}
                <ProbabilityLegend />
              </figcaption>
            </figure>

            <figure className={state.stage === "revealed" ? "revealed" : "sealed"}>
              {state.stage === "revealed" && painted.actRgba ? (
                <RasterCanvas
                  width={painted.width}
                  height={painted.height}
                  rgba={painted.actRgba}
                  label="Observed next-day active fire, native grid"
                />
              ) : (
                <div className="seal" style={{ aspectRatio: `${painted.width} / ${painted.height}` }}>
                  <span>Sealed. Make your own read of the forecast, then reveal.</span>
                </div>
              )}
              <figcaption>Observed by VIIRS on {origin?.target_date}</figcaption>
            </figure>

            {state.stage === "revealed" && painted.cmpRgba ? (
              <figure>
                <RasterCanvas
                  width={painted.width}
                  height={painted.height}
                  rgba={painted.cmpRgba}
                  label="Comparison of forecast and observed fire at the fold threshold"
                />
                <figcaption>
                  At threshold {threshold?.toFixed(3)}
                  <ComparisonLegend counts={painted.counts} />
                </figcaption>
              </figure>
            ) : null}
          </div>
        ) : (
          <div className="empty-stage">
            <p>Pick a fire and a day on the left, then load its forecast. The observed fire stays sealed until you reveal it.</p>
          </div>
        )}

        {forecast ? (
          <div className="details">
            <section aria-label="Provenance">
              <h3>Where this forecast came from</h3>
              <ProvenanceList provenance={{ ...forecast.provenance, time_safety_status: forecast.time_safety_status }} />
            </section>

            <section aria-label="Scores">
              <h3>Scores</h3>
              {state.stage === "revealed" && state.metrics ? (
                <MetricsTable metrics={state.metrics} />
              ) : (
                <p className="fine">Scores use the observed fire, so they unlock with the reveal.</p>
              )}
            </section>

            <section aria-label="Downloads">
              <h3>Downloads</h3>
              <div className="downloads">
                {prob && session ? (
                  <button
                    type="button"
                    onClick={async () =>
                      saveBlob(
                        await fetchArtifactBytes(prob.artifact.url, session.session_id),
                        `${state.eventId}_${origin?.origin_date}_probability.tif`,
                        "image/tiff",
                      )
                    }
                  >
                    Forecast GeoTIFF
                  </button>
                ) : null}
                <button
                  type="button"
                  onClick={() =>
                    saveBlob(
                      JSON.stringify(forecast, null, 2),
                      `${state.eventId}_${origin?.origin_date}_provenance.json`,
                      "application/json",
                    )
                  }
                >
                  Provenance JSON
                </button>
                {state.stage === "revealed" && actual && session ? (
                  <button
                    type="button"
                    onClick={async () =>
                      saveBlob(
                        await fetchArtifactBytes(actual.artifact.url, session.session_id),
                        `${state.eventId}_${origin?.target_date}_observed.tif`,
                        "image/tiff",
                      )
                    }
                  >
                    Observed fire GeoTIFF
                  </button>
                ) : null}
                {state.stage === "revealed" && state.metrics ? (
                  <button
                    type="button"
                    onClick={() =>
                      saveBlob(
                        JSON.stringify(state.metrics, null, 2),
                        `${state.eventId}_${origin?.origin_date}_metrics.json`,
                        "application/json",
                      )
                    }
                  >
                    Scores JSON
                  </button>
                ) : null}
              </div>
            </section>
          </div>
        ) : null}
      </section>
    </div>
  );
}
