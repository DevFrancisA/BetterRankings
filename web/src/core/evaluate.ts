// Evaluate policies on a log and run the simulated online test,
// a port of betterrankings/evaluate.py.

import {
  allEstimates,
  bootstrapIndices,
  clippedIps,
  doublyRobust,
  overlapDiagnostics,
  type Bootstrap,
  type Diagnostics,
  type Estimate,
  type EstimatorId,
} from "./estimators";
import { logSize, perPositionClicks, LogFormatError, type LogTable, type PositionRow } from "./logTable";
import { permutations } from "./perms";
import { expectedClicksRow, POLICY_BY_ID, policyRow, type PolicyId } from "./policies";
import { crossFit, loggedPrediction, type CrossFit } from "./rewardModel";
import {
  DEFAULT_WORLD,
  makeCatalog,
  makeParams,
  positionWeights,
  sampleQueries,
  trueClickRow,
  type WorldConfig,
} from "./world";

export const TARGET_COLUMN = "target_propensity";
export type TargetId = PolicyId | typeof TARGET_COLUMN;

export const DEFAULTS = {
  nQueries: 5000,
  exploration: 0.2,
  seed: 7,
  clip: 20 as number | null,
  nBoot: 200,
  nFolds: 2,
  nOnline: 20000,
};

export interface EvalOptions {
  cfg?: WorldConfig;
  exploration: number;
  clip: number | null;
  nBoot: number;
  seed: number;
  nFolds: number;
  onProgress?: (stage: string, fraction: number) => void;
}

export interface PolicyEvaluation {
  id: TargetId;
  name: string;
  estimates: Record<EstimatorId, Estimate>;
  diagnostics: Diagnostics;
  /** Kept for re-clipping and the weight histogram. */
  weights: Float64Array;
  dm: Float64Array | null;
}

export interface Evaluation {
  n: number;
  loggedMean: number;
  clip: number | null;
  perPosition: PositionRow[];
  policies: Partial<Record<TargetId, PolicyEvaluation>>;
  rewards: Float64Array;
  qLogged: Float64Array | null;
  bootstrap: Bootstrap;
}

export function evaluateLog(log: LogTable, ids: TargetId[], opts: EvalOptions): Evaluation {
  const f = log.features;
  const cfg: WorldConfig = { ...(opts.cfg ?? DEFAULT_WORLD), nSlots: f.m };
  if (f.dCtx !== cfg.dCtx || f.dItem !== cfg.dItem) {
    throw new LogFormatError(
      `Built-in policies need ${cfg.dCtx} context and ${cfg.dItem} item features; the log has ${f.dCtx} and ${f.dItem}.`,
    );
  }
  const n = logSize(log);
  const progress = opts.onProgress ?? (() => {});
  const params = makeParams(cfg);
  progress("Drawing bootstrap resamples", 0.05);
  const bs = bootstrapIndices(n, opts.nBoot, opts.seed);
  const perms = permutations(f.m);
  let fit: CrossFit | null = null;
  let qLogged: Float64Array | null = null;
  const policies: Evaluation["policies"] = {};
  const probs = new Float64Array(perms.count);

  ids.forEach((id, step) => {
    let piLogged: Float64Array;
    let dm: Float64Array | null = null;
    let name: string;
    if (id === TARGET_COLUMN) {
      if (!log.targetPropensity) throw new LogFormatError("The log has no target_propensity column.");
      piLogged = log.targetPropensity;
      name = "Your policy (target_propensity column)";
    } else {
      if (!fit) {
        fit = crossFit(f, log.clicks, opts.nFolds, opts.seed, (k) =>
          progress(`Fitting the click model, fold ${k + 1} of ${opts.nFolds}`, 0.1 + (0.3 * k) / opts.nFolds),
        );
        qLogged = loggedPrediction(fit, n, f.m);
      }
      piLogged = new Float64Array(n);
      dm = new Float64Array(n);
      const mm = f.m * f.m;
      for (let i = 0; i < n; i++) {
        policyRow(id, f, i, cfg, params, opts.exploration, probs);
        piLogged[i] = probs[0]; // columns follow the shown order, so ranking 0 is the logged one
        dm[i] = expectedClicksRow(probs, fit.q.subarray(i * mm, (i + 1) * mm), perms);
      }
      name = POLICY_BY_ID[id].name;
    }
    progress(`Estimating ${name}`, 0.4 + (0.5 * step) / ids.length);
    const w = piLogged.map((p, i) => p / log.propensity[i]);
    policies[id] = {
      id,
      name,
      estimates: allEstimates(log.rewards, log.propensity, piLogged, dm, dm ? qLogged : null, opts.clip, bs),
      diagnostics: overlapDiagnostics(w, opts.clip),
      weights: w,
      dm,
    };
  });
  let total = 0;
  for (let i = 0; i < n; i++) total += log.rewards[i];
  return {
    n,
    loggedMean: total / n,
    clip: opts.clip,
    perPosition: perPositionClicks(log),
    policies,
    rewards: log.rewards,
    qLogged,
    bootstrap: bs,
  };
}

/** Recompute the clip-dependent estimates in place for a new clip level. */
export function reclip(ev: Evaluation, clip: number | null): void {
  ev.clip = clip;
  for (const p of Object.values(ev.policies)) {
    if (!p) continue;
    p.estimates.clipped_ips = clippedIps(ev.rewards, p.weights, clip, ev.bootstrap);
    if (p.dm && ev.qLogged) p.estimates.doubly_robust = doublyRobust(ev.rewards, p.weights, p.dm, ev.qLogged, clip, ev.bootstrap);
    p.diagnostics = overlapDiagnostics(p.weights, clip);
  }
}

/** Point estimates only, for a sweep of clip levels (keeps chart axes steady). */
export function clipSweep(ev: Evaluation, clips: (number | null)[]): Partial<Record<TargetId, { clipped_ips: number[]; doubly_robust: number[] }>> {
  const out: ReturnType<typeof clipSweep> = {};
  const n = ev.n;
  for (const p of Object.values(ev.policies)) {
    if (!p) continue;
    const c1: number[] = [];
    const c2: number[] = [];
    for (const c of clips) {
      let s1 = 0;
      let s2 = 0;
      for (let i = 0; i < n; i++) {
        const wc = c === null ? p.weights[i] : Math.min(p.weights[i], c);
        s1 += wc * ev.rewards[i];
        if (p.dm && ev.qLogged) s2 += p.dm[i] + wc * (ev.rewards[i] - ev.qLogged[i]);
      }
      c1.push(s1 / n);
      c2.push(p.dm ? s2 / n : NaN);
    }
    out[p.id] = { clipped_ips: c1, doubly_robust: c2 };
  }
  return out;
}

export interface Truth {
  value: number;
  se: number;
}

/** The simulated online test: exact expected clicks of each policy on fresh traffic. */
export function groundTruth(
  cfg: WorldConfig,
  seed: number,
  exploration: number,
  ids: PolicyId[],
  nOnline: number,
): Record<string, Truth> {
  const params = makeParams(cfg);
  const catalog = makeCatalog(cfg, seed);
  const { features: f } = sampleQueries(cfg, catalog, nOnline, seed, "online");
  const perms = permutations(cfg.nSlots);
  const theta = positionWeights(cfg.nSlots, cfg.positionDecay);
  const g = new Float64Array(nOnline * cfg.nSlots * cfg.nSlots);
  const row = new Float64Array(cfg.nSlots * cfg.nSlots);
  for (let i = 0; i < nOnline; i++) {
    trueClickRow(cfg, params, f, i, theta, row);
    g.set(row, i * row.length);
  }
  const probs = new Float64Array(perms.count);
  const out: Record<string, Truth> = {};
  for (const id of ids) {
    const v = new Float64Array(nOnline);
    let s = 0;
    for (let i = 0; i < nOnline; i++) {
      policyRow(id, f, i, cfg, params, exploration, probs);
      v[i] = expectedClicksRow(probs, g.subarray(i * row.length, (i + 1) * row.length), perms);
      s += v[i];
    }
    const mean = s / nOnline;
    let ss = 0;
    for (let i = 0; i < nOnline; i++) ss += (v[i] - mean) * (v[i] - mean);
    out[id] = { value: mean, se: Math.sqrt(ss / (nOnline - 1)) / Math.sqrt(nOnline) };
  }
  return out;
}

/** Pairs of policies an estimator puts in the opposite order from the truth. */
export function discordantPairs(truth: number[], est: number[]): number {
  let bad = 0;
  for (let i = 0; i < truth.length; i++) {
    for (let j = i + 1; j < truth.length; j++) {
      if (!Number.isFinite(est[i]) || !Number.isFinite(est[j])) bad++;
      else if ((truth[i] - truth[j]) * (est[i] - est[j]) < 0) bad++;
    }
  }
  return bad;
}
