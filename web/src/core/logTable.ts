// Log tables: simulating them and reading/writing the CSV format
// (see betterrankings/logs.py for the column reference).

import { MAX_SLOTS, permutations } from "./perms";
import { policyRow } from "./policies";
import { Rng } from "./rng";
import {
  makeCatalog,
  makeParams,
  positionWeights,
  sampleQueries,
  trueRelevanceRow,
  type Features,
  type WorldConfig,
} from "./world";

export interface LogTable {
  queryId: string[];
  features: Features; // candidates in shown order
  itemIds: string[]; // (n * m)
  clicks: Uint8Array; // (n * m)
  rewards: Float64Array; // total clicks per query
  propensity: Float64Array;
  targetPropensity: Float64Array | null;
}

export function logSize(log: LogTable): number {
  return log.propensity.length;
}

export function simulateLog(cfg: WorldConfig, nQueries: number, exploration: number, seed: number): LogTable {
  if (!(exploration >= 0 && exploration <= 1)) throw new Error("exploration must be between 0 and 1");
  const params = makeParams(cfg);
  const catalog = makeCatalog(cfg, seed);
  const { features: f, ids } = sampleQueries(cfg, catalog, nQueries, seed, "log");
  const m = cfg.nSlots;
  const d = cfg.dItem;
  const perms = permutations(m);
  const theta = positionWeights(m, cfg.positionDecay);
  const uRank = new Rng(seed, "log-rankings");
  const uClick = new Rng(seed, "log-clicks");

  const shown: Features = {
    n: nQueries,
    m,
    dCtx: cfg.dCtx,
    dItem: d,
    ctx: f.ctx,
    pop: new Float64Array(nQueries * m),
    feats: new Float64Array(nQueries * m * d),
  };
  const itemIds: string[] = new Array(nQueries * m);
  const clicks = new Uint8Array(nQueries * m);
  const rewards = new Float64Array(nQueries);
  const propensity = new Float64Array(nQueries);
  const probs = new Float64Array(perms.count);
  const rel = new Float64Array(m);

  // Rankings are drawn for every query first, then clicks, matching the order
  // in which the Python version consumes its random streams.
  const chosen = new Int32Array(nQueries);
  for (let i = 0; i < nQueries; i++) {
    policyRow("current", f, i, cfg, params, exploration, probs);
    const u = uRank.next();
    let total = 0;
    const cdf = new Float64Array(perms.count);
    for (let p = 0; p < perms.count; p++) {
      total += probs[p];
      cdf[p] = total;
    }
    const thr = u * total;
    let c = 0;
    for (let p = 0; p < perms.count; p++) if (cdf[p] <= thr) c++;
    c = Math.min(c, perms.count - 1);
    chosen[i] = c;
    propensity[i] = probs[c];
    for (let k = 0; k < m; k++) {
      const slot = perms.slots[c * m + k];
      shown.pop[i * m + k] = f.pop[i * m + slot];
      for (let b = 0; b < d; b++) shown.feats[(i * m + k) * d + b] = f.feats[(i * m + slot) * d + b];
      itemIds[i * m + k] = String(ids[i * m + slot]);
    }
  }
  for (let i = 0; i < nQueries; i++) {
    trueRelevanceRow(cfg, params, shown, i, rel);
    let r = 0;
    for (let k = 0; k < m; k++) {
      const c = uClick.next() < rel[k] * theta[k] ? 1 : 0;
      clicks[i * m + k] = c;
      r += c;
    }
    rewards[i] = r;
  }
  const queryId = Array.from({ length: nQueries }, (_, i) => String(i + 1));
  return { queryId, features: shown, itemIds, clicks, rewards, propensity, targetPropensity: null };
}

// CSV -------------------------------------------------------------------------

export class LogFormatError extends Error {}

export function logColumns(m: number, dCtx: number, dItem: number, target = false): string[] {
  const cols = ["query_id"];
  for (let a = 1; a <= dCtx; a++) cols.push(`ctx_${a}`);
  for (let k = 1; k <= m; k++) cols.push(`item_${k}`);
  for (let k = 1; k <= m; k++) cols.push(`click_${k}`);
  for (let k = 1; k <= m; k++) {
    cols.push(`item_${k}_pop`);
    for (let b = 1; b <= dItem; b++) cols.push(`item_${k}_f${b}`);
  }
  cols.push("propensity");
  if (target) cols.push("target_propensity");
  return cols;
}

/** Split CSV text into rows of fields (RFC 4180 quoting, CRLF or LF, optional BOM). */
export function parseCsvRows(text: string): string[][] {
  if (text.charCodeAt(0) === 0xfeff) text = text.slice(1);
  const rows: string[][] = [];
  let row: string[] = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else quoted = false;
      } else field += ch;
    } else if (ch === '"') quoted = true;
    else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n" || ch === "\r") {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(field);
      field = "";
      if (row.length > 1 || row[0] !== "") rows.push(row);
      row = [];
    } else field += ch;
  }
  if (field !== "" || row.length) {
    row.push(field);
    if (row.length > 1 || row[0] !== "") rows.push(row);
  }
  return rows;
}

export function detectShape(columns: string[]): { m: number; dCtx: number; dItem: number } {
  const cols = new Set(columns);
  let m = 0;
  while (cols.has(`item_${m + 1}`)) m++;
  let dCtx = 0;
  while (cols.has(`ctx_${dCtx + 1}`)) dCtx++;
  let dItem = 0;
  while (cols.has(`item_1_f${dItem + 1}`)) dItem++;
  return { m, dCtx, dItem };
}

export interface ParseOptions {
  expectCtx?: number;
  expectItem?: number;
}

/** Parse and validate a log CSV. Throws LogFormatError with a plain explanation. */
export function parseLogCsv(text: string, opts: ParseOptions = {}): LogTable {
  const rows = parseCsvRows(text);
  if (rows.length === 0) throw new LogFormatError("The file is empty.");
  const header = rows[0].map((h) => h.trim());
  const body = rows.slice(1);
  if (body.length === 0) throw new LogFormatError("The file has a header row but no data rows.");
  const { m, dCtx, dItem } = detectShape(header);
  if (m < 2 || m > MAX_SLOTS)
    throw new LogFormatError(`Expected columns item_1 to item_K with K between 2 and ${MAX_SLOTS}.`);
  if (opts.expectCtx !== undefined && dCtx !== opts.expectCtx)
    throw new LogFormatError(
      `Expected ${opts.expectCtx} context columns (ctx_1 to ctx_${opts.expectCtx}), found ${dCtx}.`,
    );
  if (opts.expectItem !== undefined && dItem !== opts.expectItem)
    throw new LogFormatError(
      `Expected ${opts.expectItem} item feature columns per position (item_K_f1 to item_K_f${opts.expectItem}), found ${dItem}.`,
    );
  const col = new Map(header.map((h, i) => [h, i]));
  const missing = logColumns(m, dCtx, dItem).filter((c) => !col.has(c));
  if (missing.length) throw new LogFormatError(`Missing columns: ${missing.slice(0, 8).join(", ")}.`);
  const hasTarget = col.has("target_propensity");

  const n = body.length;
  const ctx = new Float64Array(n * dCtx);
  const pop = new Float64Array(n * m);
  const feats = new Float64Array(n * m * dItem);
  const clicks = new Uint8Array(n * m);
  const rewards = new Float64Array(n);
  const propensity = new Float64Array(n);
  const target = hasTarget ? new Float64Array(n) : null;
  const itemIds: string[] = new Array(n * m);
  const queryId: string[] = new Array(n);

  const num = (r: string[], name: string, line: number): number => {
    const raw = (r[col.get(name)!] ?? "").trim();
    const v = raw === "" ? NaN : Number(raw);
    if (!Number.isFinite(v)) throw new LogFormatError(`Row ${line}: ${name} must be a number, found "${raw}".`);
    return v;
  };
  for (let i = 0; i < n; i++) {
    const r = body[i];
    const line = i + 2;
    if (r.length < header.length)
      throw new LogFormatError(`Row ${line} has ${r.length} fields but the header has ${header.length}.`);
    queryId[i] = r[col.get("query_id")!];
    for (let a = 0; a < dCtx; a++) ctx[i * dCtx + a] = num(r, `ctx_${a + 1}`, line);
    let total = 0;
    for (let k = 0; k < m; k++) {
      itemIds[i * m + k] = r[col.get(`item_${k + 1}`)!];
      const c = num(r, `click_${k + 1}`, line);
      if (c !== 0 && c !== 1) throw new LogFormatError(`Row ${line}: click_${k + 1} must be 0 or 1, found ${c}.`);
      clicks[i * m + k] = c;
      total += c;
      pop[i * m + k] = num(r, `item_${k + 1}_pop`, line);
      for (let b = 0; b < dItem; b++) feats[(i * m + k) * dItem + b] = num(r, `item_${k + 1}_f${b + 1}`, line);
    }
    rewards[i] = total;
    const p = num(r, "propensity", line);
    if (!(p > 0 && p <= 1))
      throw new LogFormatError(`Row ${line}: propensity must be greater than 0 and at most 1, found ${p}.`);
    propensity[i] = p;
    if (target) {
      const t = num(r, "target_propensity", line);
      if (!(t >= 0 && t <= 1))
        throw new LogFormatError(`Row ${line}: target_propensity must be between 0 and 1, found ${t}.`);
      target[i] = t;
    }
  }
  return {
    queryId,
    features: { n, m, dCtx, dItem, ctx, pop, feats },
    itemIds,
    clicks,
    rewards,
    propensity,
    targetPropensity: target,
  };
}

/** Serialise a log to CSV. JavaScript's shortest round-trip number format keeps floats exact. */
export function logToCsv(log: LogTable): string {
  const { n, m, dCtx, dItem } = log.features;
  const header = logColumns(m, dCtx, dItem, log.targetPropensity !== null);
  const lines = [header.join(",")];
  for (let i = 0; i < n; i++) {
    const r: (string | number)[] = [log.queryId[i]];
    for (let a = 0; a < dCtx; a++) r.push(log.features.ctx[i * dCtx + a]);
    for (let k = 0; k < m; k++) r.push(log.itemIds[i * m + k]);
    for (let k = 0; k < m; k++) r.push(log.clicks[i * m + k]);
    for (let k = 0; k < m; k++) {
      r.push(log.features.pop[i * m + k]);
      for (let b = 0; b < dItem; b++) r.push(log.features.feats[(i * m + k) * dItem + b]);
    }
    r.push(log.propensity[i]);
    if (log.targetPropensity) r.push(log.targetPropensity[i]);
    lines.push(r.map(String).join(","));
  }
  return lines.join("\n") + "\n";
}

export interface PositionRow {
  position: number;
  impressions: number;
  clicks: number;
  click_rate: number;
}

export function perPositionClicks(log: LogTable): PositionRow[] {
  const { n, m } = log.features;
  const out: PositionRow[] = [];
  for (let k = 0; k < m; k++) {
    let c = 0;
    for (let i = 0; i < n; i++) c += log.clicks[i * m + k];
    out.push({ position: k + 1, impressions: n, clicks: c, click_rate: c / n });
  }
  return out;
}
