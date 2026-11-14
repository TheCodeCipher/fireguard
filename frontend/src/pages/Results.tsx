/**
 * Results (Role 6, added in the Prototype redesign).
 *
 * Shows exactly what Role 4's evaluation wrote (reports/phase3/*, docs/evidence/role4/*), read at
 * build time by lib/results.ts: the gate verdicts, and per fold the pooled test AUPRC of every
 * model that has been evaluated. Baselines are evaluated today; M1 and the temporal models appear
 * when scripts/run_model_comparison.py has been run with them. Nothing is estimated or filled in:
 * a model without numbers is simply not drawn. The M1 fold-2018 preview comes from
 * docs/evidence/role3/m1_fold2018_preview.json (scripts/build_m1_preview_results.py).
 */
import preview from "../../../docs/evidence/role3/m1_fold2018_preview.json";
import { FOLDS, foldFacts, MODEL_LABEL, modelsWithResults, result, strongestBaseline } from "../lib/results";

const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? null : v.toFixed(d));
const pct = (v: number) => `${(v * 100).toFixed(2)}%`;

type Scores = { auprc: number; f1: number; iou: number; brier: number };
type PreviewFire = { event_id: string; m1: Scores; distance_decay: Scores; climatology: Scores };

export function Results() {
  // PROTOTYPE (2026-11-14): only models with numbers are drawn; nothing is listed as "not run".
  const barModels = modelsWithResults();
  const fires = preview.fires as PreviewFire[];
  const mean = preview.mean_over_fires as Record<string, Scores>;
  const maxAuprc = Math.max(
    0.01,
    ...FOLDS.flatMap((f) => barModels.map((m) => result(f, m)?.pooledAuprc ?? 0)),
  );

  return (
    <div className="results">
      <header className="page-head">
        <h1>Results</h1>
        <p>
          Every number here is computed on the held-out year of each fold, on fires the model never trained on, and is
          read straight from the evaluation reports in the repository. The headline metric is AUPRC: how well the
          probabilities rank the cells that do catch fire next day, where only about 0.1 to 0.3% of cells do.
        </p>
      </header>

      <section className="card preview-card" aria-label="First M1 results">
        <h2 className="section-title" style={{ marginTop: 0 }}>First M1 results: test year 2018</h2>
        <p className="section-intro">
          M1, the spatial U-Net, trained for the 2018 fold (validation AUPRC {fmt(preview.validation_auprc)}, best
          epoch {preview.best_epoch}), scored on {preview.n_fires} of the {preview.n_test_fires_in_fold} test fires
          of 2018: the largest ones, picked by size before scoring. Baselines are scored on the same fires.
          M1 has the higher AUPRC on {preview.m1_better_auprc_than_distance_decay} of {preview.n_fires}.
        </p>
        <div className="fold-table-wrap">
          <table className="data">
            <thead>
              <tr>
                <th scope="col">Fire</th>
                <th scope="col">M1 AUPRC</th>
                <th scope="col">Distance-decay AUPRC</th>
                <th scope="col">Climatology AUPRC</th>
                <th scope="col">M1 F1</th>
                <th scope="col">Distance-decay F1</th>
              </tr>
            </thead>
            <tbody>
              {[...fires, { event_id: "Mean of the five", ...mean } as PreviewFire].map((f) => {
                const best = Math.max(f.m1.auprc, f.distance_decay.auprc, f.climatology.auprc);
                const td = (v: number) => <td className={v === best ? "best" : undefined}>{fmt(v)}</td>;
                return (
                  <tr key={f.event_id}>
                    <th scope="row">{f.event_id}</th>
                    {td(f.m1.auprc)}
                    {td(f.distance_decay.auprc)}
                    {td(f.climatology.auprc)}
                    <td>{fmt(f.m1.f1)}</td>
                    <td>{fmt(f.distance_decay.f1)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <p className="fine" style={{ marginTop: "0.6rem" }}>
          Source: docs/evidence/role3/m1_fold2018_preview.json, built by scripts/build_m1_preview_results.py from
          Role 4&apos;s per-event scores. Five fires from one test year; the full comparison covers all four.
        </p>
      </section>

      <h2 className="section-title">Test AUPRC by held-out year</h2>
      <p className="section-intro">
        Pooled over every scored cell of the test fires. Baselines on all four held-out years.
      </p>
      <div className="bars">
        {FOLDS.map((fold) => (
          <div className="bar-fold" key={fold}>
            <b>{fold}</b>
            <div className="bar-rows">
              {barModels.map((m) => {
                const r = result(fold, m);
                const v = r?.status === "OK" ? r.pooledAuprc : null;
                if (v === null) return null;
                const isBaseline = m === "distance_decay" || m === "climatology";
                return (
                  <div className="bar-row" key={m}>
                    <span className="label">{MODEL_LABEL[m] ?? m}</span>
                    <span className="bar-track" aria-hidden="true">
                      {v !== null ? (
                        <span
                          className={`bar-fill${isBaseline ? " baseline" : ""}`}
                          style={{ width: `${(v / maxAuprc) * 100}%` }}
                        />
                      ) : null}
                    </span>
                    <span className="value">{fmt(v)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      <h2 className="section-title">Per fold</h2>
      <div className="fold-table-wrap card">
        <table className="data">
          <caption style={{ padding: "0.9rem 0.85rem 0.3rem" }}>
            Source: reports/phase3/ablation_results.csv and docs/evidence/role4/baseline_results.json.
          </caption>
          <thead>
            <tr>
              <th scope="col">Test year</th>
              <th scope="col">Test fires</th>
              <th scope="col">Cells on fire next day</th>
              <th scope="col">Strongest baseline (validation)</th>
              <th scope="col">Distance-decay AUPRC</th>
              <th scope="col">Climatology AUPRC</th>
              <th scope="col">Distance-decay F1</th>
            </tr>
          </thead>
          <tbody>
            {FOLDS.map((fold) => {
              const f = foldFacts(fold);
              const dd = result(fold, "distance_decay");
              const cl = result(fold, "climatology");
              const best = Math.max(dd?.pooledAuprc ?? 0, cl?.pooledAuprc ?? 0);
              const cell = (v: number | null | undefined) => (
                <td className={v === best ? "best" : undefined}>{fmt(v) ?? "–"}</td>
              );
              return (
                <tr key={fold}>
                  <th scope="row">{fold}</th>
                  <td>
                    {f.events} <span className="fine">({f.groups} groups)</span>
                  </td>
                  <td>{pct(f.prevalence)}</td>
                  <td>{MODEL_LABEL[strongestBaseline[fold]] ?? strongestBaseline[fold]}</td>
                  {cell(dd?.pooledAuprc)}
                  {cell(cl?.pooledAuprc)}
                  <td>{fmt(dd?.pooledF1) ?? "–"}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="fine" style={{ marginTop: "0.8rem" }}>
        Bold marks the best AUPRC in each row. Paired bootstrap intervals and per-event differences are in
        reports/phase3/model_comparison.md and docs/baseline_results.md.
      </p>
    </div>
  );
}
