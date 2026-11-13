/**
 * Results (Role 6, added in the Prototype redesign).
 *
 * Shows exactly what Role 4's evaluation wrote (reports/phase3/*, docs/evidence/role4/*), read at
 * build time by lib/results.ts: the gate verdicts, and per fold the pooled test AUPRC of every
 * model that has been evaluated. Baselines are evaluated today; M1 and the temporal models appear
 * when scripts/run_model_comparison.py has been run with them. Nothing is estimated or filled in:
 * a model that has not run is shown as "not run".
 */
import {
  contextAblation,
  decisionText,
  FOLDS,
  foldFacts,
  gate2,
  gate3,
  MODEL_LABEL,
  modelsWithResults,
  result,
  strongestBaseline,
} from "../lib/results";

const fmt = (v: number | null | undefined, d = 3) => (v === null || v === undefined ? null : v.toFixed(d));
const pct = (v: number) => `${(v * 100).toFixed(2)}%`;

function pillClass(code: string) {
  if (code === "NOT_RUN") return "open";
  if (code === "FAILS" || code === "REMOVE") return "fail";
  return "done";
}

/** Models always listed in the table: the two soft baselines plus M1 (Gate 2's model). */
const TABLE_MODELS = ["distance_decay", "climatology", "m1_unet"];

export function Results() {
  const evaluated = modelsWithResults();
  const barModels = Array.from(new Set([...TABLE_MODELS, ...evaluated]));
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

      <section className="gate-cards" aria-label="Gate decisions">
        <article className="card gate-card">
          <h3>
            Gate 2 <span className={`pill ${pillClass(gate2.decision)}`}>{decisionText(gate2.decision)}</span>
          </h3>
          <p>
            M1, the spatial U-Net, against the strongest baseline of each fold (chosen on validation AUPRC). It passes
            by matching or beating it; passing unlocks training of the temporal models M3 and M4.
          </p>
          <p style={{ marginTop: "0.7rem" }}>
            The bar to beat, test AUPRC 2018 to 2021:{" "}
            <b className="num">
              {FOLDS.map((f) => fmt(result(f, strongestBaseline[f])?.pooledAuprc)).join(" / ")}
            </b>
          </p>
        </article>
        <article className="card gate-card gate-wide">
          <h3>Gate 3: what each added module is worth</h3>
          <p>Each module is kept only if its paired bootstrap interval is above zero on at least two folds.</p>
          <ul className="gate-list">
            {Object.entries(gate3).map(([pair, g]) => (
              <li key={pair}>
                <span>
                  {pair
                    .split(" vs ")
                    .map((m) => MODEL_LABEL[m] ?? m)
                    .join(" vs ")}
                  <span className="fine"> adds {g.adds}</span>
                </span>
                <span className={`pill ${pillClass(g.decision)}`}>{decisionText(g.decision)}</span>
              </li>
            ))}
            <li>
              <span>
                Three days of history against five
                <span className="fine"> for the chosen temporal model</span>
              </span>
              <span className={`pill ${pillClass(contextAblation.decision)}`}>
                {decisionText(contextAblation.decision)}
              </span>
            </li>
          </ul>
        </article>
      </section>

      <h2 className="section-title">Test AUPRC by held-out year</h2>
      <p className="section-intro">
        Pooled over every scored cell of the test fires. Grey bars are baselines; green bars are FireGuard models.
      </p>
      <div className="bars">
        {FOLDS.map((fold) => (
          <div className="bar-fold" key={fold}>
            <b>{fold}</b>
            <div className="bar-rows">
              {barModels.map((m) => {
                const r = result(fold, m);
                const v = r?.status === "OK" ? r.pooledAuprc : null;
                const isBaseline = m === "distance_decay" || m === "climatology";
                return (
                  <div className={`bar-row${v === null ? " na" : ""}`} key={m}>
                    <span className="label">{MODEL_LABEL[m] ?? m}</span>
                    <span className="bar-track" aria-hidden="true">
                      {v !== null ? (
                        <span
                          className={`bar-fill${isBaseline ? " baseline" : ""}`}
                          style={{ width: `${(v / maxAuprc) * 100}%` }}
                        />
                      ) : null}
                    </span>
                    <span className="value">{v !== null ? fmt(v) : "not run"}</span>
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
              <th scope="col">M1 AUPRC</th>
              <th scope="col">Distance-decay F1</th>
            </tr>
          </thead>
          <tbody>
            {FOLDS.map((fold) => {
              const f = foldFacts(fold);
              const dd = result(fold, "distance_decay");
              const cl = result(fold, "climatology");
              const m1 = result(fold, "m1_unet");
              const m1v = m1?.status === "OK" ? m1.pooledAuprc : null;
              const best = Math.max(dd?.pooledAuprc ?? 0, cl?.pooledAuprc ?? 0, m1v ?? 0);
              const cell = (v: number | null | undefined) =>
                v === null || v === undefined ? (
                  <td className="na">not run</td>
                ) : (
                  <td className={v === best ? "best" : undefined}>{fmt(v)}</td>
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
                  {cell(m1v)}
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
