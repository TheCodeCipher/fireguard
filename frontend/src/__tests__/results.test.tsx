/**
 * Results page (Prototype redesign): it must show only what Role 4's reports contain.
 * Baselines have numbers today; M1 has none, so it must read "not run", never a zero.
 */
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
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
  it("shows the committed baseline AUPRC and marks unrun models as not run", () => {
    render(<Results />);
    const decay2018 = result("2018", "distance_decay")!.pooledAuprc!.toFixed(3);
    expect(screen.getAllByText(decay2018).length).toBeGreaterThan(0);
    if (result("2018", "m1_unet")?.status !== "OK") {
      expect(screen.getAllByText("not run").length).toBeGreaterThan(0);
    }
    expect(screen.getAllByText(/Gate 2/).length).toBeGreaterThan(0);
  });
});
