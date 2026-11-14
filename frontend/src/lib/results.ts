/**
 * Results data for the Results and Overview pages (Role 6, added in the Prototype redesign).
 *
 * Read at BUILD time from the files Role 4's scripts write and the repository commits, so the
 * page always shows exactly what the evaluation produced and nothing is typed in by hand:
 *   reports/phase3/ablation_results.csv     pooled + event-macro metrics per fold and model
 *   reports/phase3/gate_decisions.json      Gate 2 / Gate 3 decisions (NOT_RUN until run)
 *   docs/evidence/role4/baseline_results.json  test-set sizes and fire prevalence per fold
 * After `scripts/run_model_comparison.py` is run with M1, rebuild (or reload `npm run dev`) and
 * the M1 column and the Gate 2 verdict appear on their own. Models with status NOT RUN are shown
 * as "not run", never as zero.
 */
import ablationCsv from "../../../reports/phase3/ablation_results.csv?raw";
import baselineJson from "../../../docs/evidence/role4/baseline_results.json";
import gateJson from "../../../reports/phase3/gate_decisions.json";

export const FOLDS = ["2018", "2019", "2020", "2021"] as const;

/** Display names for the model keys Role 4's comparison script writes. */
export const MODEL_LABEL: Record<string, string> = {
  distance_decay: "Distance-decay baseline",
  climatology: "Climatology baseline",
  m1_unet: "M1 spatial U-Net",
  m2_unet_no_terrain: "M2 U-Net without terrain",
  m3_stacked_unet: "M3 stacked days, T=5",
  m3_stacked_unet_T3: "M3 stacked days, T=3",
  m3_stacked_unet_base40: "M3 at M4's size",
  m4_convlstm_unet: "M4 ConvLSTM, T=5",
};

export interface ResultRow {
  fold: string;
  model: string;
  status: string; // "OK" | "NOT RUN"
  nEvents: number | null;
  pooledAuprc: number | null;
  macroAuprc: number | null;
  pooledF1: number | null;
  pooledBrier: number | null;
}

const num = (v: string | undefined) => (v === undefined || v === "" ? null : Number(v));

/** Minimal CSV reader for Role 4's file (plain comma-separated, no quoted commas). */
export function parseAblation(text: string): ResultRow[] {
  const [head, ...lines] = text.trim().split(/\r?\n/);
  const cols = head.split(",");
  const at = (cells: string[], name: string) => cells[cols.indexOf(name)];
  return lines.map((line) => {
    const c = line.split(",");
    return {
      fold: at(c, "fold"),
      model: at(c, "model"),
      status: at(c, "status"),
      nEvents: num(at(c, "n_events")),
      pooledAuprc: num(at(c, "pooled_auprc")),
      macroAuprc: num(at(c, "macro_auprc")),
      pooledF1: num(at(c, "pooled_f1")),
      pooledBrier: num(at(c, "pooled_brier")),
    };
  });
}

export const results: ResultRow[] = parseAblation(ablationCsv);

export function result(fold: string, model: string): ResultRow | undefined {
  return results.find((r) => r.fold === fold && r.model === model);
}

/** Models that have at least one evaluated fold (always the two soft baselines; more later). */
export function modelsWithResults(): string[] {
  const seen = new Set(results.filter((r) => r.status === "OK").map((r) => r.model));
  return Object.keys(MODEL_LABEL).filter((m) => seen.has(m));
}

interface BaselineFold {
  test_events: number;
  test_groups: number;
  pooled: { distance_decay: { n_pos: number; n_valid: number } };
}
const baselineFolds = (baselineJson as unknown as { folds: Record<string, BaselineFold> }).folds;

/** Test-set size and next-day fire prevalence (share of scored cells that were on fire). */
export function foldFacts(fold: string) {
  const f = baselineFolds[fold];
  const p = f.pooled.distance_decay;
  return { events: f.test_events, groups: f.test_groups, prevalence: p.n_pos / p.n_valid };
}

export interface GateDecision {
  decision: string;
  [k: string]: unknown;
}
const gates = gateJson as unknown as {
  strongest_baseline_by_validation_auprc: Record<string, string>;
  gate2: GateDecision;
  gate3: Record<string, GateDecision & { adds: string }>;
  context_ablation_T3_vs_T5: GateDecision;
};
export const gate2 = gates.gate2;
export const gate3 = gates.gate3;
export const contextAblation = gates.context_ablation_T3_vs_T5;
export const strongestBaseline = gates.strongest_baseline_by_validation_auprc;

/** Plain-language reading of a decision code. */
export function decisionText(code: string): string {
  return (
    {
      NOT_RUN: "Not run yet",
      // PROTOTYPE: a gate scored on fewer than all four folds is never shown as a pass
      INCOMPLETE: "Incomplete: not all four years trained",
      BEATS: "Passed: beats the baseline",
      MATCHES: "Passed: matches the baseline",
      FAILS: "Failed: worse than the baseline",
      RETAIN: "Keep the module",
      REMOVE: "Drop the module",
      KEEP_T5: "Keep five days of context",
      PREFER_T3: "Three days are enough",
    }[code] ?? code
  );
}
