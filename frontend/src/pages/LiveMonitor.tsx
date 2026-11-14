/**
 * Live Monitor (Role 6, P2-R6-02 / P3-R6-04): NASA FIRMS active-fire detections, observation only.
 *
 * Rules kept here:
 *  - Detections come through Role 5's backend proxy (/api/v1/fires); the FIRMS MAP_KEY stays on
 *    the server and never reaches this page.
 *  - Each satellite/product (Suomi-NPP, NOAA-20, NOAA-21; NRT or standard processing) is a
 *    separate, labelled source. They are never merged into one layer and never shown as a model
 *    target or a FireGuard prediction. No forecast is ever drawn on this map.
 *  - Requests are limited to 5-day windows, the conservative chunk size the plan fixes until
 *    the FIRMS area-API limits are re-verified (the backend itself allows up to 31 days).
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { ApiError, api } from "../api/client";
import { FIRMS_SOURCES, type FireObservation, type FirmsSource } from "../api/types";
import { MapView, type PointFeature } from "../components/MapView";

const SOURCE_LABEL: Record<FirmsSource, string> = {
  VIIRS_SNPP_NRT: "Suomi-NPP VIIRS, near real time",
  VIIRS_SNPP_SP: "Suomi-NPP VIIRS, standard processing",
  VIIRS_NOAA20_NRT: "NOAA-20 VIIRS, near real time",
  VIIRS_NOAA20_SP: "NOAA-20 VIIRS, standard processing",
  VIIRS_NOAA21_NRT: "NOAA-21 VIIRS, near real time",
  VIIRS_NOAA21_SP: "NOAA-21 VIIRS, standard processing",
};

const REGIONS: Record<string, { label: string; bbox: string }> = {
  west: { label: "Western United States", bbox: "-125,31,-102,49" },
  ca: { label: "California", bbox: "-124.5,32.5,-114,42" },
  pnw: { label: "Pacific Northwest", bbox: "-125,42,-116,49" },
  world: { label: "Whole world", bbox: "-180,-90,180,90" },
};

const MAX_DAYS = 5;

const isoDay = (d: Date) => d.toISOString().slice(0, 10);
function daysBetween(a: string, b: string) {
  return Math.round((Date.parse(b) - Date.parse(a)) / 86_400_000);
}

function formatAge(seconds: number | null): string {
  if (seconds === null) return "unknown";
  if (seconds < 3600) return `${Math.round(seconds / 60)} min`;
  if (seconds < 86_400) return `${(seconds / 3600).toFixed(1)} h`;
  return `${(seconds / 86_400).toFixed(1)} days`;
}

const escapeHtml = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

/** Popup: source, platform and acquisition time are always shown (P3-R6-04). */
function popup(p: Record<string, unknown>): string {
  const row = (k: string, v: unknown) =>
    `<tr><th>${k}</th><td>${escapeHtml(v === null || v === undefined || v === "" ? "–" : String(v))}</td></tr>`;
  return `<table class="popup">${row("Satellite", p.platform)}${row("Source", p.source)}${row(
    "Acquired (UTC)",
    `${p.acquisition_date} ${String(p.acquisition_time).replace(/(\d\d)(\d\d)/, "$1:$2")}`,
  )}${row("Confidence", p.confidence)}${row("FRP (MW)", p.frp)}${row("Day/night", p.daynight)}</table>
  <p class="popup-note">Satellite observation, not a FireGuard forecast.</p>`;
}

export function LiveMonitor() {
  const today = new Date();
  const [source, setSource] = useState<FirmsSource>("VIIRS_SNPP_NRT");
  const [end, setEnd] = useState(isoDay(today));
  const [start, setStart] = useState(isoDay(new Date(today.getTime() - 86_400_000)));
  const [region, setRegion] = useState("west");
  const [confidence, setConfidence] = useState("");
  const [obs, setObs] = useState<FireObservation[] | null>(null);
  const [meta, setMeta] = useState<string | null>(null);
  const [fixture, setFixture] = useState(false);
  const [error, setError] = useState<{ code: string; message: string } | null>(null);
  const [busy, setBusy] = useState(false);

  // Default the dates to the newest day FIRMS actually has for this source; "today" can be past
  // the end of its feed, which the backend rejects with FIRMS_DATE_UNAVAILABLE.
  const [latest, setLatest] = useState<string | null>(null);
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

  const load = useCallback(async () => {
    setBusy(true);
    setError(null);
    setFixture(false);
    try {
      const r = await api.fires({
        source,
        start,
        end,
        bbox: REGIONS[region].bbox,
        confidence: confidence || undefined,
      });
      setObs(r.observations);
      setMeta(
        `${r.total_count.toLocaleString()} detections from ${r.platform} (${r.product}), ${r.start_date} to ${r.end_date}. ` +
          `Newest detection is ${formatAge(r.data_age_seconds)} old. Retrieved ${new Date(r.retrieved_at).toLocaleString()}` +
          (r.has_more ? `; showing the first ${r.count.toLocaleString()}.` : "."),
      );
    } catch (e) {
      setObs(null);
      setMeta(null);
      setError(e instanceof ApiError ? { code: e.code, message: e.message } : { code: "CLIENT", message: String(e) });
    } finally {
      setBusy(false);
    }
  }, [source, start, end, region, confidence]);

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
      setError(e instanceof ApiError ? { code: e.code, message: e.message } : { code: "CLIENT", message: String(e) });
    } finally {
      setBusy(false);
    }
  }, []);

  const points: PointFeature[] = useMemo(
    () =>
      (obs ?? []).map((o) => ({
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

  const bbox = REGIONS[region].bbox.split(",").map(Number) as [number, number, number, number];

  return (
    <div className="monitor">
      <aside className="monitor-controls" aria-label="Detection filters">
        <div>
          <h1 className="rail-title">Live monitor</h1>
          <p className="domain-note" style={{ marginBottom: 0 }}>
          Satellite fire detections from NASA FIRMS. These are observations, not FireGuard forecasts, and each
          satellite is listed separately.
          </p>
        </div>
        <label>
          Satellite and product
          <select value={source} onChange={(e) => setSource(e.target.value as FirmsSource)}>
            {FIRMS_SOURCES.map((s) => (
              <option key={s} value={s}>
                {SOURCE_LABEL[s]}
              </option>
            ))}
          </select>
        </label>
        <label>
          Region
          <select value={region} onChange={(e) => setRegion(e.target.value)}>
            {Object.entries(REGIONS).map(([k, r]) => (
              <option key={k} value={k}>
                {r.label}
              </option>
            ))}
          </select>
        </label>
        <div className="date-pair">
          <label>
            From (UTC)
            <input type="date" value={start} max={end} onChange={(e) => setStart(e.target.value)} />
          </label>
          <label>
            To (UTC)
            <input type="date" value={end} max={latest ?? isoDay(today)} onChange={(e) => setEnd(e.target.value)} />
          </label>
        </div>
        <label>
          Detection confidence
          <select value={confidence} onChange={(e) => setConfidence(e.target.value)}>
            <option value="">Any</option>
            <option value="high">High</option>
            <option value="nominal">Nominal</option>
            <option value="low">Low</option>
          </select>
        </label>
        {rangeProblem ? <p className="error">{rangeProblem}</p> : null}
        <button type="button" className="primary" onClick={load} disabled={busy || !!rangeProblem}>
          Show detections
        </button>

        {error ? (
          <div className="error" role="alert">
            {error.code === "FIRMS_NOT_CONFIGURED" ? (
              <>
                <p>
                  The backend has no FIRMS key yet. Put <code>FIRMS_MAP_KEY=…</code> in <code>backend/.env</code> and
                  restart the API.
                </p>
                <button type="button" onClick={loadFixture}>
                  Show the development fixture instead
                </button>
              </>
            ) : (
              <p>
                {error.message} ({error.code})
              </p>
            )}
          </div>
        ) : null}
        {meta ? <p className="fine">{meta}</p> : null}

        <ul className="confidence-legend" aria-label="Point colours">
          <li><span className="dot high" /> High confidence</li>
          <li><span className="dot nominal" /> Nominal</li>
          <li><span className="dot low" /> Low</li>
          <li><span className="dot unknown" /> Not reported</li>
        </ul>
      </aside>
      <section className="stage">
        {fixture ? (
          <div className="mock-banner" role="note">
            <strong>Development fixture.</strong> Four made-up points from the backend, not NASA observations.
          </div>
        ) : null}
        <MapView
          ariaLabel="Map of satellite fire detections"
          points={points}
          fitBounds={region === "world" ? null : bbox}
          pointPopup={popup}
          watermark={fixture ? "Development fixture" : null}
        />
        <p className="attribution fine">
          Fire data: NASA FIRMS (LANCE). Times are UTC acquisition times; near-real-time products are replaced by
          standard processing later.
        </p>
      </section>
    </div>
  );
}
