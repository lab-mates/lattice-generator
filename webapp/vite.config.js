import { defineConfig } from "vite";

// base './' keeps every URL relative, so the built folder works from any
// sub-path (GitHub Pages, a university server) or the offline launcher.
export default defineConfig({
  base: "./",
  worker: { format: "es" },
  build: { target: "es2022", chunkSizeWarningLimit: 1000 },
  server: { host: true },
});
