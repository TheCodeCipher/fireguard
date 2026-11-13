/** Colour mapping, comparison classes and UTM -> lon/lat corners. */
import { describe, expect, it } from "vitest";
import { cornersBounds, rasterCorners } from "../lib/geo";
import { comparisonColor, comparisonCounts, COMPARISON_COLORS, probabilityColor, targetColor } from "../lib/raster";

describe("probabilityColor", () => {
  it("leaves invalid and near-zero cells transparent", () => {
    expect(probabilityColor(Number.NaN)[3]).toBe(0);
    expect(probabilityColor(0.005)[3]).toBe(0);
  });
  it("gets darker and more opaque as probability rises", () => {
    const lo = probabilityColor(0.1);
    const hi = probabilityColor(0.9);
    expect(hi[3]).toBeGreaterThan(lo[3]);
    expect(hi[0] + hi[1] + hi[2]).toBeLessThan(lo[0] + lo[1] + lo[2]);
  });
});

describe("targetColor", () => {
  it("shows fire, hides no-fire, greys nodata", () => {
    expect(targetColor(1)[3]).toBe(255);
    expect(targetColor(0)[3]).toBe(0);
    expect(targetColor(Number.NaN)[3]).toBeGreaterThan(0);
  });
});

describe("comparison at the fold threshold", () => {
  it("classifies hits, misses and false alarms", () => {
    expect(comparisonColor(0.9, 1, 0.5)).toEqual(COMPARISON_COLORS.hit);
    expect(comparisonColor(0.1, 1, 0.5)).toEqual(COMPARISON_COLORS.missed);
    expect(comparisonColor(0.9, 0, 0.5)).toEqual(COMPARISON_COLORS.falseAlarm);
    expect(comparisonColor(0.1, 0, 0.5)[3]).toBe(0);
  });
  it("counts only valid cells", () => {
    const prob = { width: 4, height: 1, data: new Float32Array([0.9, 0.1, 0.9, Number.NaN]) };
    const target = { width: 4, height: 1, data: new Float32Array([1, 1, 0, 1]) };
    expect(comparisonCounts(prob, target, 0.5)).toEqual({ hit: 1, missed: 1, falseAlarm: 1 });
  });
});

describe("rasterCorners", () => {
  it("projects an EPSG:32610 frame to plausible western-US lon/lat", () => {
    // Real frame of fire_24025599: 269 x 321 cells of 375 m.
    const c = rasterCorners("EPSG:32610", [375, 0, 1269000, 0, -375, 4146000], 269, 321);
    expect(c).not.toBeNull();
    const [w, s, e, n] = cornersBounds(c!);
    expect(w).toBeGreaterThan(-120);
    expect(e).toBeLessThan(-100);
    expect(s).toBeGreaterThan(30);
    expect(n).toBeLessThan(45);
    expect(n).toBeGreaterThan(s);
  });
  it("returns null for an unknown CRS rather than guessing", () => {
    expect(rasterCorners("EPSG:9999", [1, 0, 0, 0, -1, 0], 1, 1)).toBeNull();
  });
});
