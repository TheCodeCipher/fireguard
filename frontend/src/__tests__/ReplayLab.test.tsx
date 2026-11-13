/**
 * Page-level test of the Replay Lab against a fake backend. It records every request the page
 * makes and checks the leakage rules from the Phase 3/4 guide (Role 6 acceptance criteria):
 *  - the reveal button is disabled until the forecast has loaded;
 *  - no request touches the actual target, its artifact or the metrics before the reveal;
 *  - no target URL appears in the page before the reveal;
 *  - the mock banner is shown for mock replays.
 * MapLibre needs WebGL and GeoTIFF decoding needs real files, so both are replaced with stubs.
 */
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("../components/MapView", () => ({
  MapView: () => <div data-testid="map" />,
}));
vi.mock("../lib/raster", async (orig) => {
  const real = await orig<typeof import("../lib/raster")>();
  return {
    ...real,
    decodeGeoTiff: vi.fn(async () => ({ width: 2, height: 2, data: new Float32Array([0.9, 0.1, 0.4, Number.NaN]) })),
    rgbaToDataUrl: () => "data:image/png;base64,",
  };
});

import { ReplayLab } from "../pages/ReplayLab";

// jsdom has no canvas 2D context or ImageData; the pixel views only need them to exist.
beforeEach(() => {
  // @ts-expect-error minimal stand-in
  globalThis.ImageData = class {
    constructor(public data: Uint8ClampedArray, public width: number, public height: number) {}
  };
  HTMLCanvasElement.prototype.getContext = vi.fn(() => ({ putImageData: vi.fn() })) as never;
});
afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

const E = "fire_24025599";
const O = "2020-05-11";
const descriptor = (id: number, type: string) => ({
  artifact_id: id,
  artifact_type: type,
  content_type: "image/tiff",
  url: `/api/v1/artifacts/${id}`,
  checksum_sha256: "x",
  metadata: { crs: "EPSG:32610", affine_transform: [375, 0, 1269000, 0, -375, 4146000], width: 2, height: 2 },
});

function fakeBackend(mock: boolean) {
  const calls: string[] = [];
  const json = (body: unknown) => Promise.resolve(new Response(JSON.stringify(body), { status: 200 }));
  const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    calls.push(`${init?.method ?? "GET"} ${url}`);
    if (url.endsWith("/wsts/events?limit=500"))
      return json({ count: 1, total_count: 1, events: [{ event_id: E, year: 2020, p1_fold_id: "LOYO-2020", has_replay: true, source_dataset: "WSTS" }] });
    if (url.endsWith(`/replays/${E}/origins`))
      return json({
        event_id: E,
        count: 1,
        origins: [{ origin_date: O, target_date: "2020-05-12", horizon_hours: 24, protocol: mock ? "MOCK_ONLY" : "P1_LOYO", fold_id: "LOYO-2020", model_version: "m", split_name: "test", is_oof: true, registry_status: mock ? "mock" : "published", is_mock: mock }],
      });
    if (url.endsWith(`/${O}/session`)) return json({ session_id: "s1", event_id: E, origin_date: O, expires_at: "z" });
    if (url.endsWith(`/${O}/inputs`)) return json({ event_id: E, origin_date: O, input_metadata: { channel_manifest: "viirs_m11" } });
    if (url.endsWith(`/${O}/forecast`))
      return json({ event_id: E, origin_date: O, target_date: "2020-05-12", protocol: "P1_LOYO", fold_id: "LOYO-2020", model_version: "m", time_safety_status: "PASS_LEVEL_A", provenance: { model_name: "M3" }, artifacts: [descriptor(1, "probability_map")] });
    if (url.endsWith("/artifacts/1") || url.endsWith("/artifacts/2")) return Promise.resolve(new Response(new ArrayBuffer(8), { status: 200 }));
    if (url.endsWith(`/${O}/reveal`)) return json({ revealed: true, revealed_at: "now" });
    if (url.endsWith(`/${O}/actual`)) return json({ revealed: true, artifacts: [descriptor(2, "actual_target")] });
    if (url.endsWith(`/${O}/metrics`))
      return json({ metrics: [{ metric_name: "auprc", value: 0.5, threshold: null, aggregation: "forecast_day", calibrator_id: "c" }, { metric_name: "f1", value: 0.4, threshold: 0.3, aggregation: "event", calibrator_id: "c" }] });
    return Promise.resolve(new Response(JSON.stringify({ error: { code: "NOT_FOUND", message: url } }), { status: 404 }));
  });
  vi.stubGlobal("fetch", fetchMock);
  return calls;
}

const POST_REVEAL = [/\/reveal$/, /\/actual$/, /\/metrics$/, /\/artifacts\/2$/];

describe("ReplayLab", () => {
  it("keeps the target sealed until the forecast is loaded and the reveal is requested", async () => {
    const calls = fakeBackend(false);
    render(<ReplayLab />);

    fireEvent.change(await screen.findByLabelText("Fire event"), { target: { value: E } });
    fireEvent.click(await screen.findByRole("button", { name: "05-12" }));

    const load = await screen.findByRole("button", { name: /Load forecast/ });
    const reveal = screen.getByRole("button", { name: "Reveal actual fire" });
    await waitFor(() => expect(load).toBeEnabled());
    expect(reveal).toBeDisabled();

    fireEvent.click(load);
    await screen.findByText(/Where this forecast came from/);
    await waitFor(() => expect(reveal).toBeEnabled());

    // Before the reveal: nothing post-reveal was requested and no target URL is in the page.
    expect(calls.filter((c) => POST_REVEAL.some((r) => r.test(c)))).toEqual([]);
    expect(document.body.innerHTML).not.toContain("/artifacts/2");
    expect(screen.getByText(/Sealed\./)).toBeInTheDocument();
    expect(screen.queryByText("Scores computed offline; the server only reads them.")).toBeNull();

    fireEvent.click(reveal);
    await screen.findByText("Scores computed offline; the server only reads them.");
    // Reveal happened before the target and the scores were fetched.
    const idx = (r: RegExp) => calls.findIndex((c) => r.test(c));
    expect(idx(/\/reveal$/)).toBeGreaterThan(-1);
    expect(idx(/\/reveal$/)).toBeLessThan(idx(/\/actual$/));
    expect(idx(/\/reveal$/)).toBeLessThan(idx(/\/metrics$/));
    expect(idx(/\/reveal$/)).toBeLessThan(idx(/\/artifacts\/2$/));
    expect(screen.queryByText(/Sealed\./)).toBeNull();
  });

  it("labels mock replays", async () => {
    fakeBackend(true);
    render(<ReplayLab />);
    fireEvent.change(await screen.findByLabelText("Fire event"), { target: { value: E } });
    fireEvent.click(await screen.findByRole("button", { name: "05-12" }));
    expect(await screen.findByText(/Synthetic demonstration\./)).toBeInTheDocument();
  });
});
