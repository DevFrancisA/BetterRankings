// Off-policy estimators, a port of betterrankings/estimators.py.
// Each estimator is a set of per-row terms so the bootstrap only resamples rows.

import { Rng } from "./rng";

export type EstimatorId = "naive" | "ips" | "snips" | "clipped_ips" | "reward_model" | "doubly_robust";

export const ESTIMATORS: EstimatorId[] = ["naive", "ips", "snips", "clipped_ips", "reward_model", "doubly_robust"];

export const ESTIMATOR_NAMES: Record<EstimatorId, string> = {
  naive: "Naive click rate",
  ips: "Inverse propensity (IPS)",
  snips: "Self-normalised IPS",
  clipped_ips: "Clipped IPS",
  reward_model: "Reward model",
  doubly_robust: "Doubly robust",
};

export const CI_LEVEL = 0.95;

export interface Estimate {
  estimate: number;
  ci_low: number;
  ci_high: number;
  ess: number | null;
  n: number;
  available: boolean;
}

export interface Bootstrap {
  n: number;
  b: number;
  idx: Int32Array; // (b * n)
}

export function bootstrapIndices(n: number, nBoot: number, seed: number): Bootstrap {
  const r = new Rng(seed, "bootstrap");
  const idx = new Int32Array(nBoot * n);
  for (let t = 0; t < nBoot * n; t++) idx[t] = Math.floor(r.next() * n);
  return { n, b: nBoot, idx };
}

export function quantileSorted(b: number[], q: number): number {
  const h = (b.length - 1) * q;
  const lo = Math.floor(h);
  const hi = Math.min(lo + 1, b.length - 1);
  return b[lo] + (h - lo) * (b[hi] - b[lo]);
}

export function percentileCi(boots: Float64Array, level = CI_LEVEL): [number, number] {
  const b = Array.from(boots).filter(Number.isFinite).sort((x, y) => x - y);
  if (b.length === 0) return [NaN, NaN];
  const tail = (1.0 - level) / 2.0;
  return [quantileSorted(b, tail), quantileSorted(b, 1.0 - tail)];
}

function sum(a: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i];
  return s;
}

export function effectiveSampleSize(w: Float64Array): number {
  let s = 0;
  let sq = 0;
  for (let i = 0; i < w.length; i++) {
    s += w[i];
    sq += w[i] * w[i];
  }
  return sq === 0 ? 0 : (s * s) / sq;
}

export function clipWeights(w: Float64Array, clip: number | null): Float64Array {
  if (clip === null || !Number.isFinite(clip)) return w;
  const out = new Float64Array(w.length);
  for (let i = 0; i < w.length; i++) out[i] = Math.min(w[i], clip);
  return out;
}

function meanEstimate(t: Float64Array, bs: Bootstrap, ess: number | null): Estimate {
  const boots = new Float64Array(bs.b);
  for (let r = 0; r < bs.b; r++) {
    let s = 0;
    const off = r * bs.n;
    for (let i = 0; i < bs.n; i++) s += t[bs.idx[off + i]];
    boots[r] = s / bs.n;
  }
  const [lo, hi] = percentileCi(boots);
  return { estimate: sum(t) / t.length, ci_low: lo, ci_high: hi, ess, n: t.length, available: true };
}

function ratioEstimate(a: Float64Array, b: Float64Array, bs: Bootstrap, ess: number | null): Estimate {
  const total = sum(b);
  const boots = new Float64Array(bs.b);
  for (let r = 0; r < bs.b; r++) {
    let sa = 0;
    let sb = 0;
    const off = r * bs.n;
    for (let i = 0; i < bs.n; i++) {
      const j = bs.idx[off + i];
      sa += a[j];
      sb += b[j];
    }
    boots[r] = sa / sb;
  }
  const [lo, hi] = percentileCi(boots);
  return { estimate: total > 0 ? sum(a) / total : NaN, ci_low: lo, ci_high: hi, ess, n: a.length, available: true };
}

export function unavailable(n: number): Estimate {
  return { estimate: NaN, ci_low: NaN, ci_high: NaN, ess: null, n, available: false };
}

const times = (x: Float64Array, y: Float64Array) => x.map((v, i) => v * y[i]);

export function naive(rewards: Float64Array, pi: Float64Array, bs: Bootstrap): Estimate {
  return ratioEstimate(times(pi, rewards), pi, bs, effectiveSampleSize(pi));
}

export function ips(rewards: Float64Array, w: Float64Array, bs: Bootstrap): Estimate {
  return meanEstimate(times(w, rewards), bs, effectiveSampleSize(w));
}

export function snips(rewards: Float64Array, w: Float64Array, bs: Bootstrap): Estimate {
  return ratioEstimate(times(w, rewards), w, bs, effectiveSampleSize(w));
}

export function clippedIps(rewards: Float64Array, w: Float64Array, clip: number | null, bs: Bootstrap): Estimate {
  const wc = clipWeights(w, clip);
  return meanEstimate(times(wc, rewards), bs, effectiveSampleSize(wc));
}

export function rewardModel(dm: Float64Array, bs: Bootstrap): Estimate {
  return meanEstimate(dm, bs, null);
}

export function doublyRobust(
  rewards: Float64Array,
  w: Float64Array,
  dm: Float64Array,
  qLogged: Float64Array,
  clip: number | null,
  bs: Bootstrap,
): Estimate {
  const wc = clipWeights(w, clip);
  const t = dm.map((d, i) => d + wc[i] * (rewards[i] - qLogged[i]));
  return meanEstimate(t, bs, effectiveSampleSize(wc));
}

export function allEstimates(
  rewards: Float64Array,
  propensity: Float64Array,
  piLogged: Float64Array,
  dm: Float64Array | null,
  qLogged: Float64Array | null,
  clip: number | null,
  bs: Bootstrap,
): Record<EstimatorId, Estimate> {
  const w = piLogged.map((p, i) => p / propensity[i]);
  const n = rewards.length;
  return {
    naive: naive(rewards, piLogged, bs),
    ips: ips(rewards, w, bs),
    snips: snips(rewards, w, bs),
    clipped_ips: clippedIps(rewards, w, clip, bs),
    reward_model: dm ? rewardModel(dm, bs) : unavailable(n),
    doubly_robust: dm && qLogged ? doublyRobust(rewards, w, dm, qLogged, clip, bs) : unavailable(n),
  };
}

// Overlap diagnostics ---------------------------------------------------------

export const POOR_ESS_FRACTION = 0.01;
export const POOR_ESS_MIN = 30;
export const LIMITED_ESS_FRACTION = 0.1;
export const EXTREME_WEIGHT_SHARE = 0.1;
export const MEAN_WEIGHT_TOLERANCE = 0.3;

export type WarningCode = "no_support" | "poor_overlap" | "limited_overlap" | "extreme_weight" | "weights_off";

export interface OverlapWarning {
  code: WarningCode;
  level: "warn" | "info";
  message: string;
}

export interface Diagnostics {
  n: number;
  ess: number;
  ess_fraction: number;
  max_weight: number;
  max_weight_share: number;
  mean_weight: number;
  fraction_nonzero: number;
  n_clipped: number;
  warnings: OverlapWarning[];
}

const fmtInt = (x: number) => Math.round(x).toLocaleString("en-US");

export function overlapDiagnostics(w: Float64Array, clip: number | null): Diagnostics {
  const n = w.length;
  const total = sum(w);
  const ess = effectiveSampleSize(w);
  let maxW = 0;
  let nonzero = 0;
  let clipped = 0;
  for (let i = 0; i < n; i++) {
    if (w[i] > maxW) maxW = w[i];
    if (w[i] > 0) nonzero++;
    if (clip !== null && Number.isFinite(clip) && w[i] > clip) clipped++;
  }
  const share = total > 0 ? maxW / total : NaN;
  const meanW = n ? total / n : NaN;
  const warnings: OverlapWarning[] = [];
  if (total === 0) {
    warnings.push({
      code: "no_support",
      level: "warn",
      message:
        "This policy never picks any list that appears in the log, so the weighting estimators have nothing to work with.",
    });
  } else {
    if (ess < Math.max(POOR_ESS_MIN, POOR_ESS_FRACTION * n)) {
      warnings.push({
        code: "poor_overlap",
        level: "warn",
        message: `Poor overlap. The weights are worth about ${fmtInt(ess)} of ${fmtInt(n)} logged queries: the current policy rarely showed the lists this policy would show, so these estimates are unreliable.`,
      });
    } else if (ess < LIMITED_ESS_FRACTION * n) {
      warnings.push({
        code: "limited_overlap",
        level: "info",
        message: `Limited overlap. The weights are worth about ${fmtInt(ess)} of ${fmtInt(n)} logged queries, so expect wide intervals.`,
      });
    }
    if (share > EXTREME_WEIGHT_SHARE) {
      warnings.push({
        code: "extreme_weight",
        level: "warn",
        message: `One logged query carries ${Math.round(share * 100)}% of the total weight, so a single click can swing the weighted estimates.`,
      });
    }
    if (Math.abs(meanW - 1) > MEAN_WEIGHT_TOLERANCE) {
      warnings.push({
        code: "weights_off",
        level: "warn",
        message: `The weights average ${meanW.toFixed(2)} instead of about 1. Either the log rarely covers this policy's choices or the logged propensities are off.`,
      });
    }
  }
  return {
    n,
    ess,
    ess_fraction: n ? ess / n : NaN,
    max_weight: maxW,
    max_weight_share: share,
    mean_weight: meanW,
    fraction_nonzero: n ? nonzero / n : NaN,
    n_clipped: clipped,
    warnings,
  };
}
