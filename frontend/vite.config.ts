/**
 * Vite config for the FireGuard frontend (Role 6, added in the Prototype integration).
 *
 * The dev server proxies /api to the FastAPI backend, so the browser talks to one origin and
 * never needs a FIRMS key or a CORS exception. Set FIREGUARD_API_TARGET to point the proxy at
 * another backend (default: the backend's documented local address).
 */
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

const apiTarget = process.env.FIREGUARD_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: { "/api": { target: apiTarget, changeOrigin: true } },
  },
  build: {
    // MapLibre GL alone is ~1 MB minified; it is split into its own chunk so the app code stays
    // small and the map library is cached separately. The limit is raised to that size.
    chunkSizeWarningLimit: 1100,
    rollupOptions: { output: { manualChunks: { maplibre: ["maplibre-gl"], geotiff: ["geotiff"] } } },
  },
  preview: {
    port: 4173,
    proxy: { "/api": { target: apiTarget, changeOrigin: true } },
  },
  test: {
    environment: "jsdom",
    include: ["src/__tests__/**/*.test.{ts,tsx}"],
    setupFiles: ["src/test/setup.ts"],
  },
});
