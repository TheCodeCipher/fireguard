/**
 * Types mirroring the backend's Pydantic schemas (backend/src/fireguard/schemas/*.py, published as
 * docs/openapi.json). Only fields the UI reads are listed; extra fields from the API are ignored.
 * If Role 5 changes a schema, change it here in the same pull request.
 */

/** Uniform error body produced by the backend's exception handlers. */
export interface ApiErrorBody {
  error: { code: string; message: string; details?: unknown };
}

// ---------------------------------------------------------------- WSTS event catalogue

export interface WstsEvent {
  event_id: string;
  name: string | null;
  year: number | null;
  region: string | null;
  complex_group_id: string | null;
  p1_fold_id: string | null;
  source_dataset: string;
  has_replay: boolean;
}

export interface WstsEventList {
  count: number;
  total_count: number;
  events: WstsEvent[];
}

// ---------------------------------------------------------------- replay

export interface ReplayOrigin {
  origin_date: string; // YYYY-MM-DD, the newest input day (day D)
  target_date: string; // D + 1, the day being forecast
  horizon_hours: number;
  protocol: string; // P1_LOYO for real runs, MOCK_ONLY for synthetic/plumbing runs
  fold_id: string;
  model_version: string;
  split_name: string;
  is_oof: boolean;
  registry_status: string; // "published" or "mock"
  is_mock: boolean;
}

export interface ReplayOriginList {
  event_id: string;
  count: number;
  origins: ReplayOrigin[];
}

export interface ReplaySession {
  session_id: string;
  event_id: string;
  origin_date: string;
  expires_at: string;
}

export interface ReplayInputs {
  event_id: string;
  origin_date: string;
  input_metadata: Record<string, unknown>;
}

/** Allow-listed display/geospatial metadata the backend returns with each artifact. */
export interface ArtifactMetadata {
  layer_name?: string;
  units?: string;
  format?: string;
  crs?: string;
  affine_transform?: number[]; // [a, b, c, d, e, f] in rasterio order
  width?: number;
  height?: number;
  nodata?: number | null;
  valid_extent?: number[];
  display_min?: number;
  display_max?: number;
}

export interface ArtifactDescriptor {
  artifact_id: number;
  artifact_type: string; // probability_map | actual_target | metrics | ...
  content_type: string | null;
  url: string; // session-bound; fetch with the X-Replay-Session-ID header
  checksum_sha256: string | null;
  metadata: ArtifactMetadata;
}

export interface ReplayForecast {
  event_id: string;
  origin_date: string;
  target_date: string;
  protocol: string;
  fold_id: string;
  model_version: string;
  config_hash: string | null;
  channel_manifest_sha256: string | null;
  time_safety_status: string;
  provenance: Record<string, unknown>;
  artifacts: ArtifactDescriptor[];
}

export interface ReplayReveal {
  revealed: boolean;
  revealed_at: string;
}

export interface ReplayActual {
  revealed: boolean;
  artifacts: ArtifactDescriptor[];
}

export interface ReplayMetric {
  metric_name: string;
  value: number;
  threshold: number | null;
  aggregation: string; // "forecast_day" (this forecast) or "event" (all days of the event)
  calibrator_id: string | null;
}

export interface ReplayMetrics {
  metrics: ReplayMetric[];
}

// ---------------------------------------------------------------- FIRMS (observation only)

export const FIRMS_SOURCES = [
  "VIIRS_SNPP_NRT",
  "VIIRS_SNPP_SP",
  "VIIRS_NOAA20_NRT",
  "VIIRS_NOAA20_SP",
  "VIIRS_NOAA21_NRT",
  "VIIRS_NOAA21_SP",
] as const;
export type FirmsSource = (typeof FIRMS_SOURCES)[number];

export interface FireObservation {
  latitude: number;
  longitude: number;
  brightness: number | null;
  frp: number | null;
  confidence: "low" | "nominal" | "high" | "unknown";
  platform: string;
  instrument: string | null;
  acquisition_date: string;
  acquisition_time: string; // HHMM UTC
  daynight: string | null;
  source: string;
  product: string;
  source_id: string;
}

export interface FiresResponse {
  source: FirmsSource;
  platform: string;
  product: string;
  start_date: string;
  end_date: string;
  bbox: string;
  retrieved_at: string;
  data_age_seconds: number | null;
  count: number;
  total_count: number;
  has_more: boolean;
  cache: { status: string; ttl_seconds: number };
  observations: FireObservation[];
}

export interface MockFiresResponse {
  observations: FireObservation[];
  count: number;
  source: string;
  mode: string;
}
