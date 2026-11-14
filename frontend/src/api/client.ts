/**
 * Thin typed client for the FireGuard backend (Role 5's retrieve-only API).
 *
 * Rules this file enforces for the whole UI:
 *  - Every call goes to the backend (same origin through the dev proxy, or VITE_API_BASE). The
 *    browser never calls NASA FIRMS directly and never holds a FIRMS key.
 *  - Replay calls after session creation carry X-Replay-Session-ID; artifact bytes are fetched
 *    with that header (the backend checks the session's forecast/reveal state per request).
 *  - The client never builds an artifact URL itself: it only follows URLs the backend returned,
 *    so a target URL cannot exist in the page before the backend hands it out after reveal.
 */
import type {
  ApiErrorBody,
  FiresResponse,
  FirmsSource,
  LiveForecast,
  MockFiresResponse,
  ReplayActual,
  ReplayForecast,
  ReplayInputs,
  ReplayMetrics,
  ReplayOriginList,
  ReplayReveal,
  ReplaySession,
  WstsEventList,
} from "./types";

const BASE = (import.meta.env.VITE_API_BASE as string | undefined)?.replace(/\/$/, "") ?? "";

/** An error carrying the backend's machine-readable code (e.g. ACTUAL_NOT_REVEALED). */
export class ApiError extends Error {
  constructor(
    public readonly status: number,
    public readonly code: string,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init: RequestInit = {}, sessionId?: string): Promise<T> {
  const headers = new Headers(init.headers);
  if (sessionId) headers.set("X-Replay-Session-ID", sessionId);
  let response: Response;
  try {
    response = await fetch(path.startsWith("http") ? path : `${BASE}${path}`, { ...init, headers });
  } catch {
    throw new ApiError(0, "NETWORK", "The FireGuard API is not reachable. Start the backend and retry.");
  }
  if (!response.ok) {
    let code = `HTTP_${response.status}`;
    let message = response.statusText || "Request failed";
    try {
      const body = (await response.json()) as ApiErrorBody;
      code = body.error?.code ?? code;
      message = body.error?.message ?? message;
    } catch {
      /* non-JSON error body: keep the HTTP status */
    }
    throw new ApiError(response.status, code, message);
  }
  return (await response.json()) as T;
}

/** Raw bytes of a backend-provided artifact URL, fetched with the replay session header. */
export async function fetchArtifactBytes(url: string, sessionId: string): Promise<ArrayBuffer> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${url}`, { headers: { "X-Replay-Session-ID": sessionId } });
  } catch {
    throw new ApiError(0, "NETWORK", "The FireGuard API is not reachable.");
  }
  if (!response.ok) {
    let code = `HTTP_${response.status}`;
    let message = "Artifact request failed";
    try {
      const body = (await response.json()) as ApiErrorBody;
      code = body.error?.code ?? code;
      message = body.error?.message ?? message;
    } catch {
      /* ignore */
    }
    throw new ApiError(response.status, code, message);
  }
  return await response.arrayBuffer();
}

/**
 * Raw bytes of a live-forecast artifact URL returned by the backend. Live forecasts have no
 * target yet, so there is no replay session and no session header.
 */
export async function fetchLiveArtifactBytes(url: string): Promise<ArrayBuffer> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${url}`);
  } catch {
    throw new ApiError(0, "NETWORK", "The FireGuard API is not reachable.");
  }
  if (!response.ok) {
    let code = `HTTP_${response.status}`;
    let message = "Artifact request failed";
    try {
      const body = (await response.json()) as ApiErrorBody;
      code = body.error?.code ?? code;
      message = body.error?.message ?? message;
    } catch {
      /* ignore */
    }
    throw new ApiError(response.status, code, message);
  }
  return await response.arrayBuffer();
}

const enc = encodeURIComponent;

export const api = {
  health: () => request<{ status: string; environment: string; version: string }>(`/api/v1/health`),

  events: () => request<WstsEventList>(`/api/v1/wsts/events?limit=500`),
  origins: (eventId: string) => request<ReplayOriginList>(`/api/v1/replays/${enc(eventId)}/origins`),

  startSession: (eventId: string, origin: string) =>
    request<ReplaySession>(`/api/v1/replays/${enc(eventId)}/${origin}/session`, { method: "POST" }),
  inputs: (eventId: string, origin: string, sid: string) =>
    request<ReplayInputs>(`/api/v1/replays/${enc(eventId)}/${origin}/inputs`, {}, sid),
  forecast: (eventId: string, origin: string, sid: string) =>
    request<ReplayForecast>(`/api/v1/replays/${enc(eventId)}/${origin}/forecast`, {}, sid),
  reveal: (eventId: string, origin: string, sid: string) =>
    request<ReplayReveal>(`/api/v1/replays/${enc(eventId)}/${origin}/reveal`, { method: "POST" }, sid),
  actual: (eventId: string, origin: string, sid: string) =>
    request<ReplayActual>(`/api/v1/replays/${enc(eventId)}/${origin}/actual`, {}, sid),
  metrics: (eventId: string, origin: string, sid: string) =>
    request<ReplayMetrics>(`/api/v1/replays/${enc(eventId)}/${origin}/metrics`, {}, sid),

  fires: (q: {
    source: FirmsSource;
    start: string;
    end: string;
    bbox: string;
    confidence?: string;
    minFrp?: number;
  }) => {
    const p = new URLSearchParams({
      source: q.source,
      start_date: q.start,
      end_date: q.end,
      bbox: q.bbox,
      limit: "5000",
    });
    if (q.confidence) p.set("confidence", q.confidence);
    if (q.minFrp !== undefined) p.set("min_frp", String(q.minFrp));
    return request<FiresResponse>(`/api/v1/fires?${p.toString()}`);
  },
  /** Latest published live next-day forecast (404 LIVE_FORECAST_NOT_AVAILABLE if none). */
  liveForecast: () => request<LiveForecast>(`/api/v1/live/forecasts`),

  /** Backend's labelled development fixture (four synthetic points); never NASA data. */
  mockFires: () => request<MockFiresResponse>(`/api/v1/firms/observations`),
};
