// The browser port must reproduce the Python package on the shared fixtures in
// ../fixtures (written by scripts/make_fixtures.py).

import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";

import { ESTIMATORS } from "../src/core/estimators";
import { evaluateLog, groundTruth, reclip, type TargetId } from "../src/core/evaluate";
import { parseLogCsv, simulateLog } from "../src/core/logTable";
import { POLICY_IDS } from "../src/core/policies";
import { deriveSeed, Rng } from "../src/core/rng";
import { DEFAULT_WORLD } from "../src/core/world";

const FIXTURES = resolve(dirname(fileURLToPath(import.meta.url)), "../../fixtures");
const expected = JSON.parse(readFileSync(resolve(FIXTURES, "expected.json"), "utf-8"));

function close(actual: number | null, want: number | null, label: string) {
  if (want === null) {
    expect(actual === null || !Number.isFinite(actual), `${label}: expected n/a, got ${actual}`).toBe(true);
    return;
  }
  expect(actual, label).not.toBeNull();
  const tol = 1e-9 * Math.max(1, Math.abs(want));
  expect(Math.abs((actual as number) - want), `${label}: ${actual} vs ${want}`).toBeLessThanOrEqual(tol);
}

describe("random numbers", () => {
  it("matches the Python stream bit for bit", () => {
    const r = new Rng(expected.rng.seed);
    expect(Array.from(r.uniforms(8))).toEqual(expected.rng.uniforms);
    for (const d of expected.rng.derived) expect(deriveSeed(d.seed, d.name)).toBe(d.value);
    const z = new Rng(3, "normals").normals(6);
    expected.rng.normals.forEach((v: number, i: number) => close(z[i], v, `normal ${i}`));
  });
});

for (const c of expected.cases) {
  describe(`fixture ${c.name}`, () => {
    const cfg = { ...DEFAULT_WORLD, nSlots: c.config.n_slots };
    const csv = readFileSync(resolve(FIXTURES, c.log), "utf-8");
    const log = parseLogCsv(csv, { expectCtx: 3, expectItem: 3 });

    if (c.simulated) {
      it("simulates the same log", () => {
        const sim = simulateLog(cfg, c.config.n_queries, c.config.exploration, c.config.seed);
        expect(sim.itemIds).toEqual(log.itemIds);
        expect(Array.from(sim.clicks)).toEqual(Array.from(log.clicks));
        const pairs: [Float64Array, Float64Array, string][] = [
          [sim.propensity, log.propensity, "propensity"],
          [sim.features.ctx, log.features.ctx, "ctx"],
          [sim.features.pop, log.features.pop, "pop"],
          [sim.features.feats, log.features.feats, "feats"],
        ];
        for (const [a, b, name] of pairs) {
          let worst = 0;
          for (let i = 0; i < a.length; i++) worst = Math.max(worst, Math.abs(a[i] - b[i]) / Math.max(1, Math.abs(b[i])));
          expect(worst, name).toBeLessThan(1e-12);
        }
      });

      it("runs the same online test", () => {
        const truth = groundTruth(cfg, c.config.seed, c.config.exploration, POLICY_IDS, c.config.n_online);
        for (const id of POLICY_IDS) {
          close(truth[id].value, c.truth[id].value, `${id} truth`);
          close(truth[id].se, c.truth[id].se, `${id} truth se`);
        }
      });
    }

    it("gives the same estimates, intervals and diagnostics", () => {
      const ids = Object.keys(c.evaluations[0].policies) as TargetId[];
      const first = c.evaluations[0];
      const ev = evaluateLog(log, ids, {
        cfg,
        exploration: c.config.exploration,
        clip: first.clip,
        nBoot: c.config.n_boot,
        seed: c.simulated ? c.config.seed : c.config.eval_seed,
        nFolds: c.config.n_folds,
      });
      for (const want of c.evaluations) {
        if (want !== first) reclip(ev, want.clip);
        close(ev.loggedMean, want.logged_mean, "logged mean");
        want.per_position.forEach((row: Record<string, number>, k: number) => {
          expect(ev.perPosition[k].clicks).toBe(row.clicks);
          close(ev.perPosition[k].click_rate, row.click_rate, `position ${k + 1} rate`);
        });
        for (const id of ids) {
          const got = ev.policies[id]!;
          const exp = want.policies[id];
          for (const est of ESTIMATORS) {
            const g = got.estimates[est];
            const e = exp.estimates[est];
            const label = `clip ${want.clip} ${id} ${est}`;
            expect(g.available, label).toBe(e.available);
            close(g.estimate, e.estimate, `${label} estimate`);
            close(g.ci_low, e.ci_low, `${label} ci_low`);
            close(g.ci_high, e.ci_high, `${label} ci_high`);
            close(g.ess, e.ess, `${label} ess`);
          }
          const d = exp.diagnostics;
          close(got.diagnostics.ess, d.ess, `${id} diag ess`);
          close(got.diagnostics.max_weight, d.max_weight, `${id} max weight`);
          close(got.diagnostics.mean_weight, d.mean_weight, `${id} mean weight`);
          expect(got.diagnostics.n_clipped, `${id} n_clipped`).toBe(d.n_clipped);
          expect(got.diagnostics.warnings.map((w) => w.code), `${id} warnings`).toEqual(d.warning_codes);
        }
      }
    });
  });
}
