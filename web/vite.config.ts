import { defineConfig } from "vitest/config";

// Served as a GitHub Pages project site at https://<user>.github.io/BetterRankings/
export default defineConfig({
  base: "/BetterRankings/",
  build: { outDir: "dist", target: "es2022" },
  worker: { format: "es" },
  test: { include: ["test/**/*.test.ts"], testTimeout: 60000 },
});
