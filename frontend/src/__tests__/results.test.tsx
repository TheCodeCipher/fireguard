/**
 * Results page (Prototype redesign): it must show only what the evaluation files contain.
 * Baselines have numbers for all four folds; M1 shows its fold-2018 preview from the committed
 * preview file. Models without numbers are not drawn at all.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import preview from "../../../docs/evidence/role3/m1_fold2018_preview.json";
import { parseAblation, result } from "../lib/results";
import { Results } from "../pages/Results";

describe("parseAblation", () => {
  it("reads numbers and keeps NOT RUN rows empty", () => {
    const rows = parseAblation(
      "fold,model,status,n_events,pooled_auprc\n2018,a,OK,3,0.25\n2018,b,NOT RUN,,\n",
    );
    expect(rows[0].pooledAuprc).toBe(0.25);
    expect(rows[1].status).toBe("NOT RUN");
    expect(rows[1].pooledAuprc).toBeNull();
  });
});

describe("Results page", () => {
  it("shows the committed baseline AUPRC and the M1 preview, and never says not run", () => {
    render(<Results />);
    const decay2018 = result("2018", "distance_decay")!.pooledAuprc!.toFixed(3);
    expect(screen.getAllByText(decay2018).length).toBeGreaterThan(0);
    expect(screen.getAllByText(preview.mean_over_fires.m1.auprc.toFixed(3)).length).toBeGreaterThan(0);
    expect(screen.queryByText("not run")).toBeNull();
  });
});
