/**
 * Overview and Methodology pages (Role 6, P2-R6-01 / P2-R6-03 / P4-R6-04).
 *
 * Plain statements of what FireGuard is and is not, written from the project plan and the
 * current gate log. Status lines are the honest current state (no trained model yet, gates
 * pending) and must be updated when docs/phase_gate_log.md changes.
 */

export function Overview({ go }: { go: (page: "replay" | "monitor" | "method") => void }) {
  return (
    <div className="prose-page overview">
      <h1>Which burning cells will a satellite see on fire tomorrow?</h1>
      <p className="lede">
        FireGuard is a research prototype that forecasts, for a wildfire that is already burning, the probability
        that each 375 m cell will show an active-fire detection the next day. It is trained and tested on the
        WildfireSpreadTS benchmark: 607 western-US fires from 2018 to 2021.
      </p>

      <div className="two-ways">
        <button type="button" className="way" onClick={() => go("replay")}>
          <h2>Replay a past fire</h2>
          <p>
            Pick a fire and a day, load the forecast the model made without ever seeing that fire, then reveal
            what the satellite actually observed.
          </p>
        </button>
        <button type="button" className="way" onClick={() => go("monitor")}>
          <h2>See today's detections</h2>
          <p>
            Current NASA FIRMS satellite detections, by satellite. Observation only: FireGuard does not
            forecast live fires.
          </p>
        </button>
      </div>

      <h2>What it does not do</h2>
      <ul>
        <li>No live forecasts, alerts or evacuation advice.</li>
        <li>No fire perimeters: the target is a satellite detection, which clouds and smoke can hide.</li>
        <li>No claims outside the western-US benchmark it was tested on.</li>
      </ul>

      <h2>Where the project stands</h2>
      <p>
        The data pipeline, models, evaluation protocol, replay backend and this interface are integrated and run end
        to end. No model has been trained for real yet: training waits for the spatial baseline to pass its gate.
        Replays you see in this build come from a synthetic demonstration run and say so on screen.{" "}
        <button type="button" className="link" onClick={() => go("method")}>
          Read the method
        </button>
      </p>
    </div>
  );
}

export function Methodology() {
  return (
    <div className="prose-page">
      <h1>Method and data</h1>

      <h2>The forecast</h2>
      <p>
        Input: the five days up to day D on the native 375 m grid. Output: for every cell, the probability of a
        Suomi-NPP VIIRS active-fire detection on day D+1. Frames are never cropped; they are padded to a multiple
        of 32 for the network and the padding is excluded from training and scoring.
      </p>

      <h2>Inputs the models may use</h2>
      <p>
        Seven channels passed a no-look-ahead audit: VIIRS M11, I2 and I1 reflectance, slope, aspect, elevation
        and the day-D active-fire mask (contract <code>wsts_strict_input_v1</code>). Weather forecasts (GFS),
        gridMET weather, drought index, vegetation indices and the year-Y land-cover map are excluded until their
        timing is shown to be safe. Representing aspect as sine and cosine is proposed but not approved.
      </p>

      <h2>Models</h2>
      <table className="plain">
        <tbody>
          <tr>
            <th scope="row">M1</th>
            <td>Spatial U-Net on day D only (Role 2)</td>
          </tr>
          <tr>
            <th scope="row">M3</th>
            <td>U-Net with the five days stacked as channels (Role 3)</td>
          </tr>
          <tr>
            <th scope="row">M4</th>
            <td>Shared encoder with a ConvLSTM over the five days (Role 3)</td>
          </tr>
          <tr>
            <th scope="row">Baselines</th>
            <td>Persistence and a distance-decay climatology (Role 4)</td>
          </tr>
        </tbody>
      </table>

      <h2>How forecasts are tested</h2>
      <p>
        Leave-one-year-out: each year 2018–2021 is held out once, and fires that belong to the same fire complex
        never sit on both sides of a split (frozen manifest, SHA-256 <code>34e29090…</code>). Every replay comes
        from the model of the fold that held that fire out. Thresholds and calibration are fitted on validation
        fires only. The headline metric is AUPRC; F1, IoU and Brier score are reported alongside, pooled and per
        event, with event-level bootstrap intervals.
      </p>

      <h2>Why the observed fire is sealed</h2>
      <p>
        The replay shows the forecast first and keeps the next-day detections on the server until you reveal
        them. The reveal is recorded, and nothing derived from the target, scores included, is sent before it.
      </p>

      <h2>Data sources</h2>
      <ul>
        <li>
          WildfireSpreadTS v1.0 (Gerard et al., 2023), Zenodo record 8006177; loader pinned at commit{" "}
          <code>f62dc49</code>.
        </li>
        <li>NASA FIRMS VIIRS active-fire detections (Suomi-NPP, NOAA-20, NOAA-21) for the Live Monitor.</li>
        <li>HLS S30 was audited for historical coverage only; it is not a model input.</li>
      </ul>
    </div>
  );
}
