/**
 * Live forecast page against a fake backend: it shows only what the backend published, follows
 * only backend URLs, labels unvalidated / stale / incomplete runs, and never draws observations.
 * MapLibre needs WebGL and GeoTIFF decoding needs real files, so both are stubbed (as in
 * ReplayLab.test.tsx).
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

const mapProps: Array<Record<string, unknown>> = [];
vi.mock("../components/MapView", () => ({
  MapView: (props: Record<string, unknown>) => {
    mapProps.push(props);
    return <div data-testid="map" />;
  },
}));
vi.mock("../lib/raster", async (orig) => {
  const real = await orig<typeof import("../lib/raster")>();
  return {
    ...real,
    decodeGeoTiff: vi.fn(async () => ({ width: 2, height: 2, data: new Float32Array([0.9, 0.1, 0.4, Number.NaN]) })),
    rgbaToDataUrl: () => "data:image/png;base64,",
  };
});

import { LiveForecastPage } from "../pages/LiveForecast";

const RUN = "live_2026-11-01_m1_t1_v1";
const region = (id: string, nanM11 = 0.1) => ({
  region_id: id,
  center_lon: -120.2,
  center_lat: 38.4,
  bbox_wgs84: [-120.7, 37.9, -119.7, 38.9],
  height: 2,
  width: 2,
  crs: "EPSG:32610",
  transform: [375, 0, 697500, 0, -375, 4136625],
  n_detections: 25,
  nan_fraction: { viirs_m11: nanM11, viirs_i2: 0.1, viirs_i1: 0.1, binary_active_fire_mask: 0 },
  reflectance_source: "earth_engine_vnp09ga_v2",
  artifacts: {
    probability: { url: `/api/v1/live/forecasts/${RUN}/${id}/probability`, checksum_sha256: "x", file_size_bytes: 1, content_type: "image/tiff" },
    fold_spread: { url: `/api/v1/live/forecasts/${RUN}/${id}/fold_spread`, checksum_sha256: "x", file_size_bytes: 1, content_type: "image/tiff" },
  },
});
const forecast = (over: Record<string, unknown> = {}) => ({
  live_run_id: RUN,
  origin_date: "2026-11-01",
  target_date: "2026-11-02",
  horizon_hours: 24,
  generated_at: "2026-11-02T06:00:00+00:00",
  stale: false,
  stale_after_hours: 36,
  model_version: "m1_t1_v1",
  architecture: "m1_spatial_unet",
  folds: [2018, 2019, 2020, 2021],
  probability_kind: "model_sigmoid_uncalibrated_fold_mean",
  gate_status: "GATED",
  is_mock: false,
  data_contract_version: "wsts_strict_input_v1",
  firms_source: "VIIRS_SNPP_NRT",
  disclaimer: "Research next-day forecast, not an official warning.",
  regions: [region("r_38p400n_120p200w"), region("r_44p000n_112p000w")],
  ...over,
});

function backend(body: unknown, status = 200) {
  const calls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      calls.push(url);
      if (url.endsWith("/api/v1/live/forecasts")) return Promise.resolve(new Response(JSON.stringify(body), { status }));
      if (url.includes("/api/v1/live/forecasts/")) return Promise.resolve(new Response(new ArrayBuffer(8), { status: 200 }));
      return Promise.resolve(new Response("{}", { status: 404 }));
    }),
  );
  return calls;
}

beforeEach(() => {
  mapProps.length = 0;
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("Live forecast page", () => {
  it("draws one probability overlay per region from backend URLs only", async () => {
    const calls = backend(forecast());
    render(<LiveForecastPage />);
    await screen.findByText(/Forecast for/);
    await waitFor(() => expect((mapProps.at(-1)?.overlays as unknown[])?.length).toBe(2));
    expect(calls.filter((u) => u.includes("/probability"))).toHaveLength(2);
    expect(calls.every((u) => u.startsWith("/api/v1/live/"))).toBe(true);
    expect(mapProps.at(-1)?.points).toBeUndefined(); // observations are never drawn here
    expect(screen.getByText(/not an official warning/)).toBeTruthy();
    expect(screen.queryByTestId("ungated-banner")).toBeNull();
    expect(screen.queryByTestId("caution-banner")).toBeNull();
  });

  it("switches to the fold-spread layer", async () => {
    const calls = backend(forecast());
    render(<LiveForecastPage />);
    await screen.findByText(/Forecast for/);
    fireEvent.change(screen.getByLabelText("Map layer"), { target: { value: "fold_spread" } });
    await waitFor(() => expect(calls.filter((u) => u.includes("/fold_spread"))).toHaveLength(2));
    expect(screen.getByText(/not an uncertainty estimate/)).toBeTruthy();
  });

  it("marks an ungated run in red", async () => {
    backend(forecast({ gate_status: "UNGATED" }));
    render(<LiveForecastPage />);
    expect((await screen.findByTestId("ungated-banner")).textContent).toMatch(/not passed Gate 2/);
  });

  it("marks a mock run", async () => {
    backend(forecast({ gate_status: "UNGATED", is_mock: true }));
    render(<LiveForecastPage />);
    expect((await screen.findByTestId("ungated-banner")).textContent).toMatch(/Mock run/);
  });

  it("warns when the run is stale or reflectance was missing", async () => {
    backend(forecast({ stale: true, regions: [region("r_a", 1.0)] }));
    render(<LiveForecastPage />);
    const banner = await screen.findByTestId("caution-banner");
    expect(banner.textContent).toMatch(/more than 36 hours old/);
    expect(banner.textContent).toMatch(/training averages/);
    expect(screen.getByText(/surface reflectance 100% missing/)).toBeTruthy();
  });

  it("shows a calm message when nothing has been published yet", async () => {
    backend({ error: { code: "LIVE_FORECAST_NOT_AVAILABLE", message: "No live forecast has been published yet" } }, 404);
    render(<LiveForecastPage />);
    expect(await screen.findByText(/No live forecast yet/)).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
  });

  it("shows other errors with a retry", async () => {
    backend({ error: { code: "LIVE_INDEX_UNREADABLE", message: "The live forecast index is unreadable" } }, 503);
    render(<LiveForecastPage />);
    expect((await screen.findByRole("alert")).textContent).toMatch(/LIVE_INDEX_UNREADABLE/);
    expect(screen.getByRole("button", { name: "Retry" })).toBeTruthy();
  });
});
