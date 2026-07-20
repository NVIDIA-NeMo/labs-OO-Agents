import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { resolve } from "path";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": resolve(__dirname, "src"),
    },
  },
  css: {
    postcss: "./postcss.config.js",
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://localhost:5001",
      "/v1": "http://localhost:5001",
    },
  },
  build: {
    outDir: "dist",
    cssMinify: "esbuild",
  },
});
