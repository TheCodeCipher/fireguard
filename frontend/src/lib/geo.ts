/**
 * Georeferencing for display. WSTS rasters are on EPSG:32610 (UTM zone 10N, 375 m cells); the
 * map is Web Mercator. MapLibre's image source takes the four lon/lat corners of the image, so we
 * project the raster's corners with proj4 and let MapLibre warp the quad.
 *
 * This is a display approximation (a UTM rectangle is not exactly a lon/lat quad), matching Role
 * 1's note that previews are display-only and analysis uses the native EPSG:32610 raster. Pixel
 * views and metrics never go through this projection.
 */
import proj4 from "proj4";

// UTM definitions for the CRSs the WSTS archive uses (all events are on EPSG:32610).
proj4.defs("EPSG:32610", "+proj=utm +zone=10 +datum=WGS84 +units=m +no_defs");
proj4.defs("EPSG:32611", "+proj=utm +zone=11 +datum=WGS84 +units=m +no_defs");

export type LngLat = [number, number];
/** top-left, top-right, bottom-right, bottom-left (MapLibre's image-source order). */
export type Corners = [LngLat, LngLat, LngLat, LngLat];

/**
 * Corners of a raster from its affine transform [a, b, c, d, e, f] (rasterio order:
 * x = a*col + b*row + c, y = d*col + e*row + f), or null if the CRS is unknown.
 */
export function rasterCorners(
  crs: string | undefined,
  transform: number[] | undefined,
  width: number,
  height: number,
): Corners | null {
  if (!crs || !transform || transform.length < 6) return null;
  if (!proj4.defs(crs)) return null;
  const [a, b, c, d, e, f] = transform;
  const at = (col: number, row: number): LngLat => {
    const x = a * col + b * row + c;
    const y = d * col + e * row + f;
    const [lon, lat] = proj4(crs, "EPSG:4326", [x, y]);
    return [lon, lat];
  };
  return [at(0, 0), at(width, 0), at(width, height), at(0, height)];
}

/** Bounding box [west, south, east, north] of a set of corners (for fitBounds). */
export function cornersBounds(c: Corners): [number, number, number, number] {
  const lons = c.map((p) => p[0]);
  const lats = c.map((p) => p[1]);
  return [Math.min(...lons), Math.min(...lats), Math.max(...lons), Math.max(...lats)];
}
