/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

const backend = "http://localhost:8000";
const proxiedPaths = [
  "/chat",
  "/stats",
  "/anomalies",
  "/insights",
  "/transactions",
  "/health",
];

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: Object.fromEntries(proxiedPaths.map((p) => [p, backend])),
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/setupTests.ts",
  },
});
