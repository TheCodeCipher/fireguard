/**
 * Small presentational pieces of the Forecast & Replay Lab. None of them fetch data; they render
 * exactly what the page passes in.
 */
import { useEffect, useRef } from "react";
import type { ReplayMetric } from "../api/types";
import { COMPARISON_COLORS } from "../lib/raster";

/** Pixel-exact view of one raster on its native grid (one WSTS cell = one canvas pixel). */
export function RasterCanvas({
  width,
  height,
  rgba,
  label,
}: {
  width: number;
  height: number;
  rgba: Uint8ClampedArray;
  label: string;
}) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const ctx = ref.current?.getContext("2d");
    if (!ctx) return;
    ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);
  }, [rgba, width, height]);
  return (
    <canvas
      ref={ref}
      width={width}
      height={height}
      className="raster-canvas"
      role="img"
      aria-label={label}
    />
  );
}

/**
 * Banner shown on every mock replay. PROTOTYPE: worded for all three mock causes the bridge
 * (scripts/build_backend_registry.py) can record - synthetic pixels, a plumbing/smoke training
 * budget, or a run covering only part of a fold or only some of the four test years.
 */
export function MockBanner() {
  return (
    <div className="mock-banner" role="note">
      <strong>Preview model.</strong> A test run, not the final model: it shows the pipeline
      working end to end, and its forecast and scores are not final results.
    </div>
  );
}

export function ProbabilityLegend({ stretch }: { stretch?: { lo: number; hi: number } | null }) {
  const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
  const ticks = stretch
    ? [0, 0.25, 0.5, 0.75, 1].map((t) => pct(stretch.lo + t * (stretch.hi - stretch.lo)))
    : ["1%", "25%", "50%", "75%", "100%"];
  return (
    <div className="legend" aria-label="Probability legend">
      <span className="legend-title">Forecast probability of a detection tomorrow</span>
      <div className="legend-ramp" />
      <div className="legend-scale">
        {ticks.map((t, i) => (
          <span key={i}>{t}</span>
        ))}
      </div>
      <span className="legend-note">
        {stretch
          ? `Colours stretched to this forecast's own range (${pct(stretch.lo)} to ${pct(stretch.hi)}); cells below the median are left clear. Uncalibrated model output.`
          : "Below 1% is left clear. Uncalibrated model output."}
      </span>
    </div>
  );
}

export function ComparisonLegend({ counts }: { counts?: { hit: number; missed: number; falseAlarm: number } }) {
  const swatch = (c: number[]) => ({ background: `rgb(${c[0]}, ${c[1]}, ${c[2]})` });
  return (
    <ul className="comparison-legend" aria-label="Comparison legend">
      <li>
        <span className="swatch" style={swatch(COMPARISON_COLORS.hit)} /> Forecast and observed
        {counts ? <b>{counts.hit.toLocaleString()}</b> : null}
      </li>
      <li>
        <span className="swatch" style={swatch(COMPARISON_COLORS.missed)} /> Observed, not forecast
        {counts ? <b>{counts.missed.toLocaleString()}</b> : null}
      </li>
      <li>
        <span className="swatch" style={swatch(COMPARISON_COLORS.falseAlarm)} /> Forecast, not observed
        {counts ? <b>{counts.falseAlarm.toLocaleString()}</b> : null}
      </li>
    </ul>
  );
}

const METRIC_LABELS: Record<string, string> = {
  auprc: "AUPRC",
  brier: "Brier score",
  f1: "F1",
  iou: "IoU",
  n_pos: "Fire cells",
  n_valid: "Scored cells",
};
const METRIC_ORDER = ["auprc", "brier", "f1", "iou", "n_pos", "n_valid"];

/** Precomputed metrics (post-reveal only): this forecast day next to the whole event. */
export function MetricsTable({ metrics }: { metrics: ReplayMetric[] }) {
  const value = (name: string, agg: string) =>
    metrics.find((m) => m.metric_name === name && m.aggregation === agg);
  const fmt = (m?: ReplayMetric) => {
    if (!m) return "n/a";
    if (m.metric_name === "n_pos" || m.metric_name === "n_valid") return Math.round(m.value).toLocaleString();
    return m.value.toFixed(3);
  };
  const threshold = metrics.find((m) => m.threshold !== null)?.threshold;
  const calibrator = metrics.find((m) => m.calibrator_id)?.calibrator_id;
  return (
    <div className="metrics">
      <table>
        <caption>Scores computed offline; the server only reads them.</caption>
        <thead>
          <tr>
            <th scope="col">Metric</th>
            <th scope="col">This day</th>
            <th scope="col">Whole event</th>
          </tr>
        </thead>
        <tbody>
          {METRIC_ORDER.map((name) => (
            <tr key={name}>
              <th scope="row">{METRIC_LABELS[name]}</th>
              <td className={value(name, "forecast_day") ? undefined : "na"}>{fmt(value(name, "forecast_day"))}</td>
              <td className={value(name, "event") ? undefined : "na"}>{fmt(value(name, "event"))}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <p className="fine">
        n/a: not defined for this forecast (AUPRC needs at least one fire cell that day). F1 and IoU use the threshold {threshold !== undefined && threshold !== null ? threshold.toFixed(3) : "–"} fitted
        on this fold's validation events ({calibrator ?? "no calibrator id"}); the held-out year was never used to choose it.
      </p>
    </div>
  );
}

const PROVENANCE_LABELS: Array<[string, string]> = [
  ["model_name", "Model"],
  ["model_version", "Version"],
  ["protocol_id", "Protocol"],
  ["fold_id", "Fold"],
  ["held_out_group_id", "Held-out fire group"],
  ["split_name", "Split"],
  ["data_schema_version", "Input contract"],
  ["time_safety_status", "Time safety"],
  ["checkpoint_sha256", "Checkpoint SHA-256"],
  ["config_hash", "Config SHA-256"],
  ["split_manifest_sha256", "Split manifest SHA-256"],
  ["input_manifest_sha256", "Input manifest SHA-256"],
  ["channel_manifest_sha256", "Channel list SHA-256"],
];

/** Where the forecast came from, as recorded by the backend (pre-reveal safe). */
export function ProvenanceList({ provenance }: { provenance: Record<string, unknown> }) {
  return (
    <dl className="provenance">
      {PROVENANCE_LABELS.map(([key, label]) => {
        const v = provenance[key];
        if (v === undefined || v === null || v === "") return null;
        const text = String(v);
        const isHash = key.endsWith("sha256") || key === "config_hash";
        return (
          <div key={key}>
            <dt>{label}</dt>
            <dd title={text} className={isHash ? "hash" : undefined}>
              {isHash ? `${text.slice(0, 12)}…` : text}
            </dd>
          </div>
        );
      })}
    </dl>
  );
}
