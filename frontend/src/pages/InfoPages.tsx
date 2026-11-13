/**
 * Overview and Methodology pages (Role 6, P2-R6-01 / P2-R6-03 / P4-R6-04; redesigned in the
 * Prototype integration).
 *
 * Plain statements of what FireGuard is and is not. The project track on the Overview is not
 * typed by hand: gate states come from reports/phase3/gate_decisions.json (lib/results.ts), so
 * the page changes when Role 4's comparison is re-run.
 *
 * The hero figure is REAL observation data, not a model output: VIIRS active-fire detections of
 * WSTS event fire_25294989 on four days of July 2021 (band 23 of the WSTS GeoTIFFs), cropped to
 * the main burn area (3,552 of the event's 3,560 detected cells), rendered by the script quoted in
 * frontend/README.md. No forecast appears on this page.
 */
import type { Page } from "../App";
import day1 from "../assets/fire_25294989_2021-07-07.png";
import day2 from "../assets/fire_25294989_2021-07-08.png";
import day3 from "../assets/fire_25294989_2021-07-10.png";
import day4 from "../assets/fire_25294989_2021-07-12.png";
import { decisionText, gate2, gate3 } from "../lib/results";

const FIRE_DAYS = [
  { src: day1, date: "7 Jul", cells: 165 },
  { src: day2, date: "8 Jul", cells: 704 },
  { src: day3, date: "10 Jul", cells: 1001 },
  { src: day4, date: "12 Jul", cells: 1167 },
];

function FireDays() {
  return (
    <figure className="card fire-days" style={{ margin: 0 }}>
      <ol>
        {FIRE_DAYS.map((d) => (
          <li key={d.date}>
            <img src={d.src} alt={`Active-fire detections on ${d.date} 2021: ${d.cells} cells`} />
            <span className="day-label">
              <b>{d.date}</b>
              <span>{d.cells.toLocaleString()} cells</span>
            </span>
          </li>
        ))}
      </ol>
      <figcaption>
        <span>
          One real fire: VIIRS detections of WSTS fire_25294989, July 2021, on 375 m cells. Cropped to the main burn
          area; counts are for the whole frame.
        </span>
        <span className="key">
          <i style={{ background: "#7d3b35" }} /> detected that day
        </span>
        <span className="key">
          <i style={{ background: "#f0bd57" }} /> detected earlier
        </span>
      </figcaption>
    </figure>
  );
}

export function Overview({ go }: { go: (page: Page) => void }) {
  const g2 = gate2.decision;
  const g2Done = g2 === "BEATS" || g2 === "MATCHES";
  const g3Run = Object.values(gate3).some((g) => g.decision !== "NOT_RUN");
  const stages = [
    {
      title: "Data frozen",
      status: "Done",
      done: true,
      text: "607 fires, leave-one-year-out split with fire complexes kept together. Manifest hash recorded.",
    },
    {
      title: "Baselines",
      status: "Done",
      done: true,
      text: "Persistence, distance-decay and climatology scored on every fold. They set the bar to beat.",
    },
    {
      title: "Gate 2: spatial model",
      status: decisionText(g2),
      done: g2Done,
      fail: g2 === "FAILS",
      text: "M1 must match or beat the strongest baseline before any temporal model is trained.",
    },
    {
      title: "Gate 3: time",
      status: g3Run ? "Decided" : "Not run yet",
      done: g3Run,
      text: "Do five days of history (M3) or a ConvLSTM (M4) add skill over M1?",
    },
    {
      title: "Real replays",
      status: "After Gate 3",
      done: false,
      text: "The chosen model's held-out forecasts, published to the Replay lab.",
    },
  ];

  return (
    <div className="overview">
      <section className="hero">
        <div>
          <h1>Which cells of a burning fire will a satellite see tomorrow?</h1>
          <p className="lede">
            FireGuard forecasts, for a wildfire that is already burning, the probability that each 375 m cell shows
            an active-fire detection the next day. It is built and tested on WildfireSpreadTS: 607 western-US fires
            from 2018 to 2021.
          </p>
          <div className="actions">
            <a
              className="button primary"
              href="#replay"
              onClick={(e) => {
                e.preventDefault();
                go("replay");
              }}
            >
              Open the replay lab
            </a>
            <a
              className="button"
              href="#results"
              onClick={(e) => {
                e.preventDefault();
                go("results");
              }}
            >
              See the results
            </a>
          </div>
        </div>
        <FireDays />
      </section>

      <h2 className="section-title">Where the project stands</h2>
      <p className="section-intro">
        Each stage has to pass before the next one starts, so nothing downstream is built on an unproven step.
      </p>
      <ol className="track">
        {stages.map((s) => (
          <li key={s.title} className={s.done ? "is-done" : undefined}>
            <h3>{s.title}</h3>
            <span className={`pill ${s.fail ? "fail" : s.done ? "done" : "open"}`}>{s.status}</span>
            <p>{s.text}</p>
          </li>
        ))}
      </ol>

      <div className="facts">
        <section>
          <h3>Replay a past fire</h3>
          <p>
            Pick a fire and a day, look at the forecast the model made without ever training on that fire, then reveal
            what the satellite actually saw and how the forecast scored.
          </p>
          <button type="button" className="link" onClick={() => go("replay")}>
            Go to the replay lab
          </button>
        </section>
        <section>
          <h3>What it does not do</h3>
          <ul>
            <li>No live forecasts, alerts or evacuation advice.</li>
            <li>No fire perimeters: the target is a satellite detection, which clouds and smoke can hide.</li>
            <li>No claims outside the western-US benchmark it was tested on.</li>
          </ul>
        </section>
        <section>
          <h3>Today's detections</h3>
          <p>
            The Live monitor shows current NASA FIRMS satellite detections, one satellite at a time. It is observation
            only: FireGuard does not forecast live fires.
          </p>
          <button type="button" className="link" onClick={() => go("monitor")}>
            Open the live monitor
          </button>
        </section>
      </div>
    </div>
  );
}

export function Methodology() {
  return (
    <div className="prose-page">
      <h1>Method and data</h1>
      <p className="lede">
        How a forecast is made, what it may look at, and how it is tested so the scores mean what they say.
      </p>

      <h2>The forecast</h2>
      <p>
        Input: the days up to day D on the native 375 m grid. Output: for every cell, the probability of a Suomi-NPP
        VIIRS active-fire detection on day D+1. Frames are never cropped; they are padded to a multiple of 32 for the
        network, and the padding is excluded from training and scoring.
      </p>

      <h2>Inputs the models may use</h2>
      <p>
        Seven channels passed a no-look-ahead audit: VIIRS M11, I2 and I1 reflectance, slope, aspect, elevation and the
        day-D active-fire mask (contract <code>wsts_strict_input_v1</code>). Weather forecasts (GFS), gridMET weather,
        drought index, vegetation indices and the year-Y land-cover map are excluded until their timing is shown to be
        safe. Representing aspect as sine and cosine is proposed but not approved.
      </p>

      <h2>Models</h2>
      <table className="plain">
        <tbody>
          <tr>
            <th scope="row">M1</th>
            <td>Spatial U-Net on day D only. Same network as M3 with one day of input, so M1 vs M3 isolates the effect of history.</td>
          </tr>
          <tr>
            <th scope="row">M3</th>
            <td>U-Net with the five days stacked as channels.</td>
          </tr>
          <tr>
            <th scope="row">M4</th>
            <td>Shared encoder with a ConvLSTM over the five days.</td>
          </tr>
          <tr>
            <th scope="row">Baselines</th>
            <td>Persistence, a distance-decay model and a climatology, fitted on validation fires only.</td>
          </tr>
        </tbody>
      </table>

      <h2>How forecasts are tested</h2>
      <p>
        Leave-one-year-out: each year 2018 to 2021 is held out once, and fires that belong to the same fire complex
        never sit on both sides of a split (frozen manifest, SHA-256 <code>34e29090…</code>). Every replay comes from
        the model of the fold that held that fire out. Thresholds and calibration are fitted on validation fires only.
        The headline metric is AUPRC; F1, IoU and Brier score are reported alongside, pooled and per event, with
        event-level bootstrap intervals.
      </p>

      <h2>Why the observed fire is sealed</h2>
      <p>
        The replay shows the forecast first and keeps the next-day detections on the server until you reveal them. The
        reveal is recorded, and nothing derived from the target, scores included, is sent before it.
      </p>

      <h2>Colours</h2>
      <p>
        Forecast probability uses Fabio Crameri's <em>lajolla</em> scientific colour map, which is perceptually uniform
        and readable with colour-vision deficiencies. Observed fire and the hit, miss and false-alarm classes use
        Okabe-Ito colours, chosen to stay distinct for colour-blind readers.
      </p>

      <h2>Data sources</h2>
      <ul>
        <li>
          WildfireSpreadTS v1.0 (Gerard et al., 2023), Zenodo record 8006177; loader pinned at commit <code>f62dc49</code>.
        </li>
        <li>NASA FIRMS VIIRS active-fire detections (Suomi-NPP, NOAA-20, NOAA-21) for the Live monitor.</li>
        <li>Basemap: CARTO Positron with Esri World Hillshade; map data © OpenStreetMap contributors.</li>
        <li>HLS S30 was audited for historical coverage only; it is not a model input.</li>
      </ul>
    </div>
  );
}
