// The simulated ranking world, a port of betterrankings/world.py.
//   P(click on item j at position k) = theta_k * sigmoid(popWeight*pop_j + x'Wv_j + baseLogit)
//   theta_k = 1 / k^positionDecay

import { Rng } from "./rng";

export interface WorldConfig {
  nItems: number;
  nSlots: number;
  dCtx: number;
  dItem: number;
  positionDecay: number;
  popWeight: number;
  baseLogit: number;
  worldSeed: number;
}

export const DEFAULT_WORLD: WorldConfig = {
  nItems: 200,
  nSlots: 4,
  dCtx: 3,
  dItem: 3,
  positionDecay: 1.0,
  popWeight: 0.6,
  baseLogit: -1.0,
  worldSeed: 1729,
};

export interface WorldParams {
  /** Each (dCtx * dItem), row-major. */
  wTrue: Float64Array;
  wModel: Float64Array;
  wLog: Float64Array;
}

export interface Catalog {
  pop: Float64Array;
  feats: Float64Array; // (nItems * dItem)
}

/** Per-query inputs with candidates in slot order. */
export interface Features {
  n: number;
  m: number;
  dCtx: number;
  dItem: number;
  ctx: Float64Array; // (n * dCtx)
  pop: Float64Array; // (n * m)
  feats: Float64Array; // (n * m * dItem)
}

export function makeParams(cfg: WorldConfig): WorldParams {
  const r = new Rng(cfg.worldSeed, "params");
  const d = cfg.dCtx * cfg.dItem;
  const wTrue = r.normals(d);
  const n1 = r.normals(d);
  const wModel = new Float64Array(d);
  for (let i = 0; i < d; i++) wModel[i] = wTrue[i] + 0.35 * n1[i];
  const n2 = r.normals(d);
  const wLog = new Float64Array(d);
  for (let i = 0; i < d; i++) wLog[i] = 0.5 * wTrue[i] + 0.5 * n2[i];
  return { wTrue, wModel, wLog };
}

export function makeCatalog(cfg: WorldConfig, seed: number): Catalog {
  const r = new Rng(seed, "catalog");
  const pop = r.normals(cfg.nItems);
  const feats = r.normals(cfg.nItems * cfg.dItem);
  return { pop, feats };
}

/** Contexts plus distinct candidates per query (partial Fisher-Yates over the catalog). */
export function sampleQueries(
  cfg: WorldConfig,
  catalog: Catalog,
  n: number,
  seed: number,
  prefix: string,
): { features: Features; ids: Int32Array } {
  const m = cfg.nSlots;
  const N = cfg.nItems;
  const ctx = new Rng(seed, `${prefix}-contexts`).normals(n * cfg.dCtx);
  const rc = new Rng(seed, `${prefix}-candidates`);
  const ids = new Int32Array(n * m);
  const pop = new Float64Array(n * m);
  const feats = new Float64Array(n * m * cfg.dItem);
  const base = new Int32Array(N);
  for (let j = 0; j < N; j++) base[j] = j;
  const deck = new Int32Array(N);
  for (let i = 0; i < n; i++) {
    deck.set(base);
    for (let k = 0; k < m; k++) {
      const j = k + Math.floor(rc.next() * (N - k));
      const held = deck[k];
      deck[k] = deck[j];
      deck[j] = held;
    }
    for (let k = 0; k < m; k++) {
      const item = deck[k];
      ids[i * m + k] = item;
      pop[i * m + k] = catalog.pop[item];
      for (let b = 0; b < cfg.dItem; b++) feats[(i * m + k) * cfg.dItem + b] = catalog.feats[item * cfg.dItem + b];
    }
  }
  return { features: { n, m, dCtx: cfg.dCtx, dItem: cfg.dItem, ctx, pop, feats }, ids };
}

/** x' W v for every candidate of query i, written into out (length m). */
export function bilinearRow(f: Features, i: number, w: Float64Array, out: Float64Array): void {
  const { dCtx, dItem, m } = f;
  const xw = new Float64Array(dItem);
  for (let b = 0; b < dItem; b++) {
    let s = 0;
    for (let a = 0; a < dCtx; a++) s += f.ctx[i * dCtx + a] * w[a * dItem + b];
    xw[b] = s;
  }
  for (let k = 0; k < m; k++) {
    let s = 0;
    const off = (i * m + k) * dItem;
    for (let b = 0; b < dItem; b++) s += f.feats[off + b] * xw[b];
    out[k] = s;
  }
}

export function sigmoid(z: number): number {
  if (z >= 0) return 1 / (1 + Math.exp(-z));
  const ez = Math.exp(z);
  return ez / (1 + ez);
}

export function positionWeights(m: number, decay: number): Float64Array {
  const t = new Float64Array(m);
  for (let k = 0; k < m; k++) t[k] = 1 / Math.pow(k + 1, decay);
  return t;
}

/** True relevance of each candidate of query i. */
export function trueRelevanceRow(cfg: WorldConfig, params: WorldParams, f: Features, i: number, out: Float64Array): void {
  bilinearRow(f, i, params.wTrue, out);
  for (let k = 0; k < f.m; k++) out[k] = sigmoid(cfg.popWeight * f.pop[i * f.m + k] + out[k] + cfg.baseLogit);
}

/** Exact click probability of slot j at position k for query i, flat (m * m) slot-major. */
export function trueClickRow(
  cfg: WorldConfig,
  params: WorldParams,
  f: Features,
  i: number,
  theta: Float64Array,
  out: Float64Array,
): void {
  const rel = new Float64Array(f.m);
  trueRelevanceRow(cfg, params, f, i, rel);
  for (let j = 0; j < f.m; j++) for (let k = 0; k < f.m; k++) out[j * f.m + k] = rel[j] * theta[k];
}
