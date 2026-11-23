/**
 * The Forecast & Replay Lab state machine (P3-R6-03 / P4-R6-03).
 *
 * The guide fixes the order: choose event -> choose forecast day -> load inputs/metadata ->
 * load the out-of-fold forecast -> reveal the actual next-day fires -> compare and read metrics.
 * This reducer is the single place that order is enforced in the UI. It is pure (no fetching) so
 * the rules are unit-tested without a browser:
 *
 *  - REVEAL is ignored unless the forecast is loaded (the backend enforces the same rule; the UI
 *    simply never offers an action the backend would refuse).
 *  - Choosing a different event or day clears every later stage: forecast, actual, metrics and
 *    reveal state never leak from one replay into the next (P4-R6-03 "reset on change").
 *  - A session-expired error drops back to "day chosen" so the user can start a fresh session.
 */
import type {
  ReplayActual,
  ReplayForecast,
  ReplayInputs,
  ReplayMetric,
  ReplayOrigin,
  ReplaySession,
} from "../api/types";

export type Stage =
  | "chooseEvent"
  | "chooseDay"
  | "ready" // session open, inputs metadata shown, forecast not yet loaded
  | "forecast" // forecast loaded; reveal available
  | "revealed"; // target and metrics unlocked

export interface ReplayState {
  stage: Stage;
  eventId: string | null;
  origin: ReplayOrigin | null;
  session: ReplaySession | null;
  inputs: ReplayInputs | null;
  forecast: ReplayForecast | null;
  actual: ReplayActual | null;
  metrics: ReplayMetric[] | null;
  busy: boolean;
  error: { code: string; message: string } | null;
}

export const initialReplayState: ReplayState = {
  stage: "chooseEvent",
  eventId: null,
  origin: null,
  session: null,
  inputs: null,
  forecast: null,
  actual: null,
  metrics: null,
  busy: false,
  error: null,
};

export type ReplayAction =
  | { type: "selectEvent"; eventId: string }
  | { type: "selectOrigin"; origin: ReplayOrigin }
  | { type: "busy" }
  | { type: "sessionReady"; session: ReplaySession; inputs: ReplayInputs }
  | { type: "forecastLoaded"; forecast: ReplayForecast }
  | { type: "revealed"; actual: ReplayActual; metrics: ReplayMetric[] }
  | { type: "failed"; code: string; message: string };

/** Everything after the chosen day, cleared together. */
const CLEARED_AFTER_DAY = {
  session: null,
  inputs: null,
  forecast: null,
  actual: null,
  metrics: null,
} as const;

export function replayReducer(state: ReplayState, action: ReplayAction): ReplayState {
  switch (action.type) {
    case "selectEvent":
      return { ...initialReplayState, stage: "chooseDay", eventId: action.eventId };
    case "selectOrigin":
      if (!state.eventId) return state;
      return {
        ...state,
        ...CLEARED_AFTER_DAY,
        stage: "chooseDay",
        origin: action.origin,
        busy: false,
        error: null,
      };
    case "busy":
      return { ...state, busy: true, error: null };
    case "sessionReady":
      if (!state.origin) return state;
      return { ...state, stage: "ready", session: action.session, inputs: action.inputs, busy: false };
    case "forecastLoaded":
      if (state.stage !== "ready" && state.stage !== "forecast") return state;
      return { ...state, stage: "forecast", forecast: action.forecast, busy: false };
    case "revealed":
      // The guard that matters: no reveal without a loaded forecast.
      if (state.stage !== "forecast" || !state.forecast) return state;
      return { ...state, stage: "revealed", actual: action.actual, metrics: action.metrics, busy: false };
    case "failed": {
      const error = { code: action.code, message: action.message };
      if (action.code === "REPLAY_SESSION_EXPIRED") {
        return { ...state, ...CLEARED_AFTER_DAY, stage: "chooseDay", busy: false, error };
      }
      return { ...state, busy: false, error };
    }
  }
}

export const canLoadForecast = (s: ReplayState) => s.stage === "ready" && !s.busy;
export const canReveal = (s: ReplayState) => s.stage === "forecast" && !!s.forecast && !s.busy;

/** True when anything about the selected replay says synthetic / mock (shown as a banner). */
export function isMockReplay(s: ReplayState): boolean {
  return (
    !!s.origin &&
    (s.origin.is_mock || s.origin.registry_status === "mock" || s.origin.protocol === "MOCK_ONLY")
  );
}
