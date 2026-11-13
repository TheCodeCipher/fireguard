/** Rules of the replay state machine (forecast first, reveal after, reset on change). */
import { describe, expect, it } from "vitest";
import type { ReplayForecast, ReplayOrigin } from "../api/types";
import {
  canLoadForecast,
  canReveal,
  initialReplayState,
  isMockReplay,
  replayReducer,
  type ReplayState,
} from "../replay/replayState";

const origin: ReplayOrigin = {
  origin_date: "2020-05-11",
  target_date: "2020-05-12",
  horizon_hours: 24,
  protocol: "P1_LOYO",
  fold_id: "LOYO-2020",
  model_version: "m",
  split_name: "test",
  is_oof: true,
  registry_status: "published",
  is_mock: false,
};
const session = { session_id: "s1", event_id: "e1", origin_date: "2020-05-11", expires_at: "x" };
const inputs = { event_id: "e1", origin_date: "2020-05-11", input_metadata: {} };
const forecast = { artifacts: [], provenance: {} } as unknown as ReplayForecast;
const actual = { revealed: true, artifacts: [] };

function at(stage: "ready" | "forecast"): ReplayState {
  let s = replayReducer(initialReplayState, { type: "selectEvent", eventId: "e1" });
  s = replayReducer(s, { type: "selectOrigin", origin });
  s = replayReducer(s, { type: "sessionReady", session, inputs });
  if (stage === "forecast") s = replayReducer(s, { type: "forecastLoaded", forecast });
  return s;
}

describe("replayReducer", () => {
  it("does not reveal before the forecast is loaded", () => {
    const s = at("ready");
    expect(canReveal(s)).toBe(false);
    const after = replayReducer(s, { type: "revealed", actual, metrics: [] });
    expect(after.stage).toBe("ready");
    expect(after.actual).toBeNull();
    expect(after.metrics).toBeNull();
  });

  it("reveals only after the forecast", () => {
    const s = at("forecast");
    expect(canLoadForecast(s)).toBe(false);
    expect(canReveal(s)).toBe(true);
    const after = replayReducer(s, { type: "revealed", actual, metrics: [] });
    expect(after.stage).toBe("revealed");
  });

  it("clears forecast, actual and metrics when the day changes", () => {
    let s = replayReducer(at("forecast"), { type: "revealed", actual, metrics: [] });
    s = replayReducer(s, { type: "selectOrigin", origin: { ...origin, origin_date: "2020-05-12" } });
    expect(s.stage).toBe("chooseDay");
    expect([s.session, s.forecast, s.actual, s.metrics]).toEqual([null, null, null, null]);
  });

  it("clears everything when the event changes", () => {
    const s = replayReducer(at("forecast"), { type: "selectEvent", eventId: "e2" });
    expect(s).toEqual({ ...initialReplayState, stage: "chooseDay", eventId: "e2" });
  });

  it("drops back to choosing the day when the session expires", () => {
    const s = replayReducer(at("forecast"), {
      type: "failed",
      code: "REPLAY_SESSION_EXPIRED",
      message: "expired",
    });
    expect(s.stage).toBe("chooseDay");
    expect(s.forecast).toBeNull();
    expect(s.error?.code).toBe("REPLAY_SESSION_EXPIRED");
  });

  it("flags mock replays from any of the backend's mock markers", () => {
    const s = at("ready");
    expect(isMockReplay(s)).toBe(false);
    expect(isMockReplay({ ...s, origin: { ...origin, is_mock: true } })).toBe(true);
    expect(isMockReplay({ ...s, origin: { ...origin, protocol: "MOCK_ONLY" } })).toBe(true);
    expect(isMockReplay({ ...s, origin: { ...origin, registry_status: "mock" } })).toBe(true);
  });
});
