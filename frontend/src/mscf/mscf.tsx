/**
 * MSCF ("mischief") / simulate-output module — an add-on layered on top of the Live Monitor.
 *
 * Everything for this mode lives in this file. The Live Monitor and MapView only call into it
 * through small optional hooks; with the mode off (the default) they behave exactly as before.
 *
 * What it does: fabricates "forecast" dots in the browser. They are NOT FireGuard model output
 * and NOT NASA FIRMS detections, so they stay identifiable wherever they are seen: their own
 * colour/size layer on the map, a per-dot popup, a persistent badge + watermark while on, and an
 * `MSCF ACTIVE` line in the console.
 */
import { useCallback, useRef, useState } from "react";
import type { PointFeature } from "../components/MapView";

export const MSCF_COLOR = "#c026d3";
export const MSCF_RING = "#f5d0fe";

type Bbox = [number, number, number, number];

export function makeSimPoints(bbox: Bbox, n = 14): PointFeature[] {
  const [w, s, e, nth] = bbox;
  // Inset 20% so the fabricated points sit inside the region rather than on its edges.
  const iw = w + (e - w) * 0.2;
  const ie = e - (e - w) * 0.2;
  const is = s + (nth - s) * 0.2;
  const inn = nth - (nth - s) * 0.2;
  const now = new Date().toISOString();
  return Array.from({ length: n }, () => ({
    lon: iw + Math.random() * (ie - iw),
    lat: is + Math.random() * (inn - is),
    properties: { probability: Math.round(Math.random() * 100) / 100, generated_at: now },
  }));
}

const esc = (s: string) =>
  s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

/** Popup for a fabricated dot: says what it is on the point itself. */
export function simPopup(p: Record<string, unknown>): string {
  const row = (k: string, v: unknown) => `<tr><th>${k}</th><td>${esc(v == null ? "–" : String(v))}</td></tr>`;
  return `<table class="popup">${row("Type", "SIMULATED dot")}${row("Reported probability", p.probability)}${row(
    "Generated",
    p.generated_at,
  )}</table>
  <p class="popup-note"><strong>Simulated output — fabricated in the browser. NOT a FireGuard model prediction and
  NOT a satellite detection.</strong></p>`;
}

export interface Mscf {
  on: boolean;
  toggle: () => void;
  /** Fabricated points to draw (empty when off). */
  points: PointFeature[];
  /** Map watermark while on, otherwise null so the page's own watermark applies. */
  watermark: string | null;
}

export function useMscf(bbox: Bbox): Mscf {
  const [on, setOn] = useState(false);
  const [points, setPoints] = useState<PointFeature[]>([]);
  const bboxRef = useRef(bbox);
  bboxRef.current = bbox;

  const toggle = useCallback(() => {
    setOn((was) => {
      const next = !was;
      if (next) {
        // eslint-disable-next-line no-console
        console.log("%cMSCF ACTIVE", `color:${MSCF_COLOR};font-weight:bold;font-size:14px`);
        // eslint-disable-next-line no-console
        console.warn(
          "MSCF ACTIVE — simulate/mischief mode is ON. The magenta dots on the Live Monitor are fabricated " +
            "in the browser. They are NOT FireGuard model predictions and NOT NASA FIRMS detections.",
        );
        setPoints(makeSimPoints(bboxRef.current));
      } else {
        // eslint-disable-next-line no-console
        console.log("MSCF off — simulated dots cleared.");
        setPoints([]);
      }
      return next;
    });
  }, []);

  return { on, toggle, points: on ? points : [], watermark: on ? "SIMULATED OUTPUT" : null };
}

/** Toggle button + legend entry, placed in the Live Monitor's control rail. */
export function MscfControls({ mscf }: { mscf: Mscf }) {
  return (
    <div className="mscf-controls">
      {mscf.on ? (
        <p className="fine" style={{ display: "flex", alignItems: "center", gap: "0.4rem", margin: 0 }}>
          <span
            aria-hidden
            style={{
              display: "inline-block",
              width: 12,
              height: 12,
              borderRadius: "50%",
              background: MSCF_COLOR,
              boxShadow: `0 0 0 2px ${MSCF_RING}`,
            }}
          />
          Simulated (fabricated)
        </p>
      ) : null}
      <button
        type="button"
        onClick={mscf.toggle}
        aria-pressed={mscf.on}
        style={{
          marginTop: "0.5rem",
          borderColor: mscf.on ? MSCF_COLOR : undefined,
          color: mscf.on ? MSCF_COLOR : undefined,
          fontWeight: mscf.on ? 600 : undefined,
        }}
      >
        {mscf.on ? "Stop simulating output" : "Simulate output"}
      </button>
    </div>
  );
}

/** Persistent on-stage badge, shown for as long as the mode is on. */
export function MscfBadge({ mscf }: { mscf: Mscf }) {
  if (!mscf.on) return null;
  return (
    <div
      role="note"
      style={{
        background: "#fdf4ff",
        border: `1px solid ${MSCF_COLOR}`,
        color: "#86198f",
        padding: "0.5rem 0.75rem",
        borderRadius: 6,
        fontSize: "0.85rem",
      }}
    >
      <strong>Simulated output active.</strong> The magenta dots are fabricated in the browser — not FireGuard
      model predictions and not NASA FIRMS detections.
    </div>
  );
}
