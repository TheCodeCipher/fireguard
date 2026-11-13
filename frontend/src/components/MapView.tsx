/**
 * MapLibre GL map used by the Replay Lab (raster overlays) and the Live Monitor (FIRMS points).
 *
 * The component is declarative: the parent passes the overlays/points it wants shown and this
 * component adds, updates and removes MapLibre sources to match. It never fetches anything
 * itself, so what is on the map is always exactly what the page decided to show (for the replay:
 * the target overlay simply is not passed in until after reveal).
 */
import maplibregl, { type GeoJSONSource, type ImageSource, type StyleSpecification } from "maplibre-gl";
import "maplibre-gl/dist/maplibre-gl.css";
import { useEffect, useRef } from "react";
import type { Corners } from "../lib/geo";

export interface RasterOverlay {
  id: string; // stable id per layer kind, e.g. "probability", "actual", "comparison"
  dataUrl: string; // PNG data URL painted from the decoded GeoTIFF
  corners: Corners;
  opacity: number;
  visible: boolean;
}

export interface PointFeature {
  lon: number;
  lat: number;
  properties: Record<string, string | number | null>;
}

interface Props {
  overlays?: RasterOverlay[];
  points?: PointFeature[];
  /** [west, south, east, north]; the map refits when this value changes. */
  fitBounds?: [number, number, number, number] | null;
  /** Text shown in the corner of the map (e.g. the mock watermark). */
  watermark?: string | null;
  /** Builds the popup HTML for a clicked point (Live Monitor). */
  pointPopup?: (p: Record<string, unknown>) => string;
  ariaLabel: string;
}

const userStyle = import.meta.env.VITE_BASEMAP_STYLE as string | undefined;

/** Desaturated OSM basemap so the data colours carry the meaning, not the background. */
const DEFAULT_STYLE: StyleSpecification = {
  version: 8,
  sources: {
    osm: {
      type: "raster",
      tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
      tileSize: 256,
      attribution: "© OpenStreetMap contributors",
      maxzoom: 18,
    },
  },
  layers: [
    { id: "background", type: "background", paint: { "background-color": "#dfe5e9" } },
    {
      id: "osm",
      type: "raster",
      source: "osm",
      paint: { "raster-saturation": -0.75, "raster-opacity": 0.85, "raster-contrast": -0.1 },
    },
  ],
};

const POINTS_SOURCE = "firms-points";

export function MapView({ overlays = [], points, fitBounds, watermark, pointPopup, ariaLabel }: Props) {
  const container = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const loaded = useRef(false);
  const pending = useRef<() => void>(() => {});

  // Create the map once.
  useEffect(() => {
    if (!container.current) return;
    const map = new maplibregl.Map({
      container: container.current,
      style: userStyle || DEFAULT_STYLE,
      center: [-119.5, 39],
      zoom: 4.3,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.addControl(new maplibregl.ScaleControl({ unit: "metric" }), "bottom-left");
    // "style.load" fires once the style is parsed, before tiles arrive. Waiting for "load"
    // instead would hold every overlay back until all basemap tiles loaded (or forever when the
    // tile server is unreachable), and the forecast must show even without a basemap.
    map.once("style.load", () => {
      loaded.current = true;
      pending.current();
    });
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
      loaded.current = false;
    };
  }, []);

  // Sync raster overlays and points whenever the inputs change.
  useEffect(() => {
    const apply = () => {
      const map = mapRef.current;
      if (!map) return;
      const wanted = new Set(overlays.map((o) => `ov-${o.id}`));
      // Remove overlays that are no longer wanted (e.g. after switching event or day).
      for (const layer of map.getStyle().layers ?? []) {
        if (layer.id.startsWith("ov-") && !wanted.has(layer.id)) {
          map.removeLayer(layer.id);
          map.removeSource(layer.id);
        }
      }
      for (const o of overlays) {
        const id = `ov-${o.id}`;
        const source = map.getSource(id) as ImageSource | undefined;
        if (source) {
          source.updateImage({ url: o.dataUrl, coordinates: o.corners });
        } else {
          map.addSource(id, { type: "image", url: o.dataUrl, coordinates: o.corners });
          map.addLayer({
            id,
            type: "raster",
            source: id,
            // Nearest-neighbour: one WSTS cell stays one visible square (no invented detail).
            paint: { "raster-resampling": "nearest", "raster-fade-duration": 0 },
          });
        }
        map.setPaintProperty(id, "raster-opacity", o.opacity);
        map.setLayoutProperty(id, "visibility", o.visible ? "visible" : "none");
      }

      if (points) {
        const data: GeoJSON.FeatureCollection = {
          type: "FeatureCollection",
          features: points.map((p) => ({
            type: "Feature",
            geometry: { type: "Point", coordinates: [p.lon, p.lat] },
            properties: p.properties,
          })),
        };
        const src = map.getSource(POINTS_SOURCE) as GeoJSONSource | undefined;
        if (src) src.setData(data);
        else {
          map.addSource(POINTS_SOURCE, { type: "geojson", data });
          map.addLayer({
            id: POINTS_SOURCE,
            type: "circle",
            source: POINTS_SOURCE,
            paint: {
              "circle-radius": ["interpolate", ["linear"], ["zoom"], 3, 2.5, 9, 6],
              // Colour by detection confidence; observations only, never a forecast colour ramp.
              "circle-color": [
                "match",
                ["get", "confidence"],
                "high", "#8f1d0e",
                "nominal", "#e8590c",
                "low", "#f8aa4b",
                "#7a8794",
              ],
              "circle-stroke-color": "#1b2836",
              "circle-stroke-width": 0.6,
            },
          });
          map.on("click", POINTS_SOURCE, (e) => {
            const f = e.features?.[0];
            if (!f || !pointPopup) return;
            new maplibregl.Popup({ closeButton: true, maxWidth: "280px" })
              .setLngLat(e.lngLat)
              .setHTML(pointPopup(f.properties as Record<string, unknown>))
              .addTo(map);
          });
          map.on("mouseenter", POINTS_SOURCE, () => (map.getCanvas().style.cursor = "pointer"));
          map.on("mouseleave", POINTS_SOURCE, () => (map.getCanvas().style.cursor = ""));
        }
      }
    };
    pending.current = apply;
    if (loaded.current) apply();
  }, [overlays, points, pointPopup]);

  // Refit only when the requested bounds actually change (not on every overlay update).
  const boundsKey = fitBounds ? fitBounds.map((v) => v.toFixed(4)).join(",") : "";
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !fitBounds) return;
    const fit = () =>
      map.fitBounds(
        [
          [fitBounds[0], fitBounds[1]],
          [fitBounds[2], fitBounds[3]],
        ],
        { padding: 40, duration: 600, maxZoom: 10 },
      );
    if (loaded.current) fit();
    else map.once("style.load", fit);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [boundsKey]);

  return (
    <div className="map-frame">
      <div ref={container} className="map" role="region" aria-label={ariaLabel} />
      {watermark ? <div className="map-watermark">{watermark}</div> : null}
    </div>
  );
}
