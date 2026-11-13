/**
 * Raster helpers: decode the backend's GeoTIFF artifacts and turn them into pixels.
 *
 * Decoding happens in the browser with geotiff.js, so the backend keeps serving the exact files
 * Role 3 exported and Role 4 verified (no server-side rendering, no second copy to keep in sync).
 */
import { fromArrayBuffer } from "geotiff";

/** One decoded single-band raster on its native WSTS grid. */
export interface DecodedRaster {
  width: number;
  height: number;
  data: Float32Array; // row-major, NaN where the cell is invalid / nodata
}

/**
 * Decode band 1 of a GeoTIFF. Integer rasters with a nodata value (the actual-target raster
 * uses 255) are converted so nodata becomes NaN; everything downstream treats NaN as "no data".
 */
export async function decodeGeoTiff(bytes: ArrayBuffer, nodata?: number | null): Promise<DecodedRaster> {
  const tiff = await fromArrayBuffer(bytes);
  const image = await tiff.getImage();
  const [band] = (await image.readRasters({ samples: [0] })) as unknown as ArrayLike<number>[];
  const width = image.getWidth();
  const height = image.getHeight();
  const fileNodata = image.getGDALNoData();
  const nd = nodata ?? fileNodata ?? null;
  const data = new Float32Array(width * height);
  for (let i = 0; i < data.length; i++) {
    const v = band[i];
    data[i] = nd !== null && v === nd ? Number.NaN : v;
  }
  return { width, height, data };
}

// ------------------------------------------------------------------------- colour maps

type RGBA = [number, number, number, number];

/**
 * Probability ramp: transparent below 1 %, then pale straw -> orange -> deep ember. Sequential,
 * monotone in lightness, so it reads correctly in greyscale and for colour-blind viewers.
 */
const PROB_STOPS: Array<[number, [number, number, number]]> = [
  [0.0, [252, 231, 166]],
  [0.25, [248, 170, 75]],
  [0.5, [232, 89, 12]],
  [0.75, [176, 42, 12]],
  [1.0, [110, 18, 8]],
];

export function probabilityColor(p: number): RGBA {
  if (!Number.isFinite(p)) return [0, 0, 0, 0];
  if (p < 0.01) return [0, 0, 0, 0];
  const x = Math.min(1, Math.max(0, p));
  for (let i = 1; i < PROB_STOPS.length; i++) {
    const [x1, c1] = PROB_STOPS[i];
    const [x0, c0] = PROB_STOPS[i - 1];
    if (x <= x1) {
      const t = (x - x0) / (x1 - x0);
      return [
        Math.round(c0[0] + t * (c1[0] - c0[0])),
        Math.round(c0[1] + t * (c1[1] - c0[1])),
        Math.round(c0[2] + t * (c1[2] - c0[2])),
        Math.round(140 + 115 * x), // fainter at low probability so the basemap stays readable
      ];
    }
  }
  return [110, 18, 8, 255];
}

/** Actual target: detected fire cells solid ink-red, no-fire transparent, nodata hatched grey. */
export function targetColor(v: number): RGBA {
  if (!Number.isFinite(v)) return [120, 130, 140, 90];
  return v >= 1 ? [143, 29, 14, 255] : [0, 0, 0, 0];
}

/**
 * Forecast-vs-target comparison at the fold's validation threshold (the same threshold the
 * F1/IoU metrics use): hit = both, missed = target only, false alarm = forecast only.
 * Hue-separated (teal / ember / violet) so the three classes do not rely on red-green.
 */
export const COMPARISON_COLORS = {
  hit: [29, 106, 114, 255] as RGBA,
  missed: [232, 89, 12, 255] as RGBA,
  falseAlarm: [124, 92, 186, 255] as RGBA,
};

export function comparisonColor(p: number, y: number, threshold: number): RGBA {
  if (!Number.isFinite(p) || !Number.isFinite(y)) return [0, 0, 0, 0];
  const predicted = p >= threshold;
  const observed = y >= 1;
  if (predicted && observed) return COMPARISON_COLORS.hit;
  if (observed) return COMPARISON_COLORS.missed;
  if (predicted) return COMPARISON_COLORS.falseAlarm;
  return [0, 0, 0, 0];
}

/** Count the comparison classes (shown next to the comparison view). */
export function comparisonCounts(prob: DecodedRaster, target: DecodedRaster, threshold: number) {
  let hit = 0;
  let missed = 0;
  let falseAlarm = 0;
  for (let i = 0; i < prob.data.length; i++) {
    const p = prob.data[i];
    const y = target.data[i];
    if (!Number.isFinite(p) || !Number.isFinite(y)) continue;
    const predicted = p >= threshold;
    if (predicted && y >= 1) hit++;
    else if (y >= 1) missed++;
    else if (predicted) falseAlarm++;
  }
  return { hit, missed, falseAlarm };
}

/** Paint a raster into RGBA bytes with a per-pixel colour function. */
export function toImageData(
  width: number,
  height: number,
  color: (i: number) => RGBA,
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(width * height * 4);
  for (let i = 0; i < width * height; i++) {
    const c = color(i);
    out[i * 4] = c[0];
    out[i * 4 + 1] = c[1];
    out[i * 4 + 2] = c[2];
    out[i * 4 + 3] = c[3];
  }
  return out;
}

/** Render RGBA bytes to a PNG data URL (used as a MapLibre image source). */
export function rgbaToDataUrl(width: number, height: number, rgba: Uint8ClampedArray): string {
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  if (!ctx) return "";
  ctx.putImageData(new ImageData(new Uint8ClampedArray(rgba), width, height), 0, 0);
  return canvas.toDataURL("image/png");
}
