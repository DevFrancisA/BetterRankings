// Cross-fitted logistic click model, a port of betterrankings/reward_model.py.
// Features: [1, pop, v, x, x_a*v_b, position one-hot (positions 2..K)].

import { Rng } from "./rng";
import { sigmoid, type Features } from "./world";

export const RIDGE = 1.0;
export const NEWTON_STEPS = 10;

export function nFeatures(dCtx: number, dItem: number, m: number): number {
  return 2 + dItem + dCtx + dCtx * dItem + (m - 1);
}

/** Feature vector for query i, slot j shown at 0-based position k. */
function designRow(f: Features, i: number, j: number, k: number, out: Float64Array): void {
  const { m, dCtx, dItem } = f;
  out.fill(0);
  let c = 0;
  out[c++] = 1;
  out[c++] = f.pop[i * m + j];
  const vo = (i * m + j) * dItem;
  for (let b = 0; b < dItem; b++) out[c++] = f.feats[vo + b];
  for (let a = 0; a < dCtx; a++) out[c++] = f.ctx[i * dCtx + a];
  for (let a = 0; a < dCtx; a++) for (let b = 0; b < dItem; b++) out[c++] = f.ctx[i * dCtx + a] * f.feats[vo + b];
  if (k > 0) out[c + k - 1] = 1;
}

function dot(a: Float64Array, b: Float64Array): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

/** Solve H x = g for symmetric positive definite H (Cholesky). */
export function solveSpd(h: Float64Array, g: Float64Array, d: number): Float64Array {
  const l = new Float64Array(d * d);
  for (let i = 0; i < d; i++) {
    for (let j = 0; j <= i; j++) {
      let s = h[i * d + j];
      for (let k = 0; k < j; k++) s -= l[i * d + k] * l[j * d + k];
      l[i * d + j] = i === j ? Math.sqrt(Math.max(s, 1e-300)) : s / l[j * d + j];
    }
  }
  const y = new Float64Array(d);
  for (let i = 0; i < d; i++) {
    let s = g[i];
    for (let k = 0; k < i; k++) s -= l[i * d + k] * y[k];
    y[i] = s / l[i * d + i];
  }
  const x = new Float64Array(d);
  for (let i = d - 1; i >= 0; i--) {
    let s = y[i];
    for (let k = i + 1; k < d; k++) s -= l[k * d + i] * x[k];
    x[i] = s / l[i * d + i];
  }
  return x;
}

/** Ridge logistic regression by a fixed number of Newton steps on (query, position) rows. */
export function fitLogistic(f: Features, clicks: Uint8Array, rows: Int32Array): Float64Array {
  const m = f.m;
  const d = nFeatures(f.dCtx, f.dItem, m);
  const nr = rows.length * m;
  const x = new Float64Array(nr * d);
  const y = new Float64Array(nr);
  const tmp = new Float64Array(d);
  let r = 0;
  for (const i of rows) {
    for (let k = 0; k < m; k++) {
      designRow(f, i, k, k, tmp);
      x.set(tmp, r * d);
      y[r] = clicks[i * m + k];
      r++;
    }
  }
  const beta = new Float64Array(d);
  const xi = new Float64Array(d);
  for (let step = 0; step < NEWTON_STEPS; step++) {
    const h = new Float64Array(d * d);
    const g = new Float64Array(d);
    for (let r2 = 0; r2 < nr; r2++) {
      const row = x.subarray(r2 * d, (r2 + 1) * d);
      xi.set(row);
      const p = sigmoid(dot(xi, beta));
      const w = p * (1 - p);
      const resid = y[r2] - p;
      for (let a = 0; a < d; a++) {
        const xa = xi[a];
        if (xa === 0) continue;
        g[a] += xa * resid;
        const wa = w * xa;
        for (let b = 0; b <= a; b++) h[a * d + b] += wa * xi[b];
      }
    }
    for (let a = 0; a < d; a++) {
      for (let b = 0; b < a; b++) h[b * d + a] = h[a * d + b];
      const pen = a === 0 ? 0 : RIDGE;
      h[a * d + a] += pen;
      g[a] -= pen * beta[a];
    }
    const delta = solveSpd(h, g, d);
    for (let a = 0; a < d; a++) beta[a] = beta[a] + delta[a];
  }
  return beta;
}

export function foldIds(n: number, nFolds: number, seed: number): Int32Array {
  const r = new Rng(seed, "folds");
  const out = new Int32Array(n);
  for (let i = 0; i < n; i++) out[i] = Math.floor(r.next() * nFolds);
  return out;
}

export interface CrossFit {
  /** Held-out predicted click probability, flat (n * m * m): slot j at position k. */
  q: Float64Array;
  folds: Int32Array;
  coefs: Float64Array[];
}

export function crossFit(
  f: Features,
  clicks: Uint8Array,
  nFolds: number,
  seed: number,
  onFold?: (k: number) => void,
): CrossFit {
  const { n, m } = f;
  const folds = foldIds(n, nFolds, seed);
  const q = new Float64Array(n * m * m);
  const coefs: Float64Array[] = [];
  const d = nFeatures(f.dCtx, f.dItem, m);
  const tmp = new Float64Array(d);
  for (let k = 0; k < nFolds; k++) {
    onFold?.(k);
    const held: number[] = [];
    let train: number[] = [];
    for (let i = 0; i < n; i++) (folds[i] === k ? held : train).push(i);
    if (train.length === 0) train = Array.from({ length: n }, (_, i) => i);
    const beta = fitLogistic(f, clicks, Int32Array.from(train));
    coefs.push(beta);
    for (const i of held) {
      for (let j = 0; j < m; j++) {
        for (let pos = 0; pos < m; pos++) {
          designRow(f, i, j, pos, tmp);
          q[(i * m + j) * m + pos] = sigmoid(dot(tmp, beta));
        }
      }
    }
  }
  return { q, folds, coefs };
}

/** Predicted clicks of the ranking that was shown (slot k at position k). */
export function loggedPrediction(cf: CrossFit, n: number, m: number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    for (let k = 0; k < m; k++) s += cf.q[(i * m + k) * m + k];
    out[i] = s;
  }
  return out;
}
