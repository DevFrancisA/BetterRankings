// Ranking policies, a port of betterrankings/policies.py. Each policy gives a
// probability to every ranking of a query's candidates.

import { permIndex, permutations, type Perms } from "./perms";
import { bilinearRow, type Features, type WorldConfig, type WorldParams } from "./world";

export const LOGGING_TEMPERATURE = 1.0;
export const LOGGING_POP_WEIGHT = 0.8;
export const SOFT_TEMPERATURE = 0.5;
export const NOISE_EXTRA_EXPLORATION = 0.25;
export const NOISE_TEMPERATURE_FACTOR = 2.0;

export type PolicyId = "current" | "popularity" | "relevance" | "relevance_soft" | "noisy_logging" | "reversed";

export interface PolicyInfo {
  id: PolicyId;
  name: string;
  short: string;
  description: string;
}

export const POLICIES: PolicyInfo[] = [
  {
    id: "current",
    name: "Current policy (logging)",
    short: "Current",
    description: "The policy that collected the log. A sanity check: every estimator should recover it.",
  },
  {
    id: "popularity",
    name: "Popularity ranker",
    short: "Popularity",
    description: "Sorts candidates by global popularity, ignoring the query.",
  },
  {
    id: "relevance",
    name: "Relevance model ranker",
    short: "Relevance",
    description: "Sorts by a learned relevance model that is close to, but not exactly, the truth.",
  },
  {
    id: "relevance_soft",
    name: "Soft relevance ranker",
    short: "Soft relevance",
    description: "Samples rankings from the relevance model with some randomness.",
  },
  {
    id: "noisy_logging",
    name: "Noisier logging policy",
    short: "Noisier logging",
    description: "The logging policy with more exploration and a flatter score distribution.",
  },
  {
    id: "reversed",
    name: "Reversed relevance ranker",
    short: "Reversed",
    description: "Puts the least relevant items first. Deliberately worse.",
  },
];
export const POLICY_IDS: PolicyId[] = POLICIES.map((p) => p.id);
export const POLICY_BY_ID = Object.fromEntries(POLICIES.map((p) => [p.id, p])) as Record<PolicyId, PolicyInfo>;

export function noisyExploration(exploration: number): number {
  return Math.min(1, exploration + NOISE_EXTRA_EXPLORATION);
}

/** Plackett-Luce probability of every ranking, written into out (length perms.count). */
export function plackettLuce(scores: Float64Array, temperature: number, perms: Perms, out: Float64Array): void {
  const m = perms.m;
  const s = new Float64Array(m);
  for (let j = 0; j < m; j++) s[j] = scores[j] / temperature;
  const sp = new Float64Array(m);
  for (let p = 0; p < perms.count; p++) {
    for (let k = 0; k < m; k++) sp[k] = s[perms.slots[p * m + k]];
    let logp = 0;
    for (let k = 0; k < m - 1; k++) {
      let mx = -Infinity;
      for (let t = k; t < m; t++) if (sp[t] > mx) mx = sp[t];
      let sum = 0;
      for (let t = k; t < m; t++) sum += Math.exp(sp[t] - mx);
      logp += sp[k] - (mx + Math.log(sum));
    }
    out[p] = Math.exp(logp);
  }
}

/** All mass on the ranking sorting scores high to low; ties keep the lower slot first. */
export function greedy(scores: Float64Array, out: Float64Array): void {
  const m = scores.length;
  const order = Array.from({ length: m }, (_, j) => j);
  order.sort((a, b) => (scores[a] > scores[b] ? -1 : scores[a] < scores[b] ? 1 : a - b));
  out.fill(0);
  out[permIndex(order)] = 1;
}

function mixture(scores: Float64Array, temperature: number, epsilon: number, perms: Perms, out: Float64Array): void {
  plackettLuce(scores, temperature, perms, out);
  for (let p = 0; p < perms.count; p++) out[p] = (1 - epsilon) * out[p] + epsilon / perms.count;
}

/** Probabilities over rankings for query i under a policy. */
export function policyRow(
  id: PolicyId,
  f: Features,
  i: number,
  cfg: WorldConfig,
  params: WorldParams,
  exploration: number,
  out: Float64Array,
): void {
  const m = f.m;
  const perms = permutations(m);
  const s = new Float64Array(m);
  const pop = f.pop.subarray(i * m, (i + 1) * m);
  const relevance = () => {
    bilinearRow(f, i, params.wModel, s);
    for (let k = 0; k < m; k++) s[k] = cfg.popWeight * pop[k] + s[k];
  };
  const logging = () => {
    bilinearRow(f, i, params.wLog, s);
    for (let k = 0; k < m; k++) s[k] = LOGGING_POP_WEIGHT * pop[k] + s[k];
  };
  switch (id) {
    case "current":
      logging();
      return mixture(s, LOGGING_TEMPERATURE, exploration, perms, out);
    case "noisy_logging":
      logging();
      return mixture(s, LOGGING_TEMPERATURE * NOISE_TEMPERATURE_FACTOR, noisyExploration(exploration), perms, out);
    case "popularity":
      s.set(pop);
      return greedy(s, out);
    case "relevance":
      relevance();
      return greedy(s, out);
    case "relevance_soft":
      relevance();
      return plackettLuce(s, SOFT_TEMPERATURE, perms, out);
    case "reversed":
      relevance();
      for (let k = 0; k < m; k++) s[k] = -s[k];
      return greedy(s, out);
  }
}

/** Expected clicks when rankings follow probs, given a (m * m) slot-by-position click matrix. */
export function expectedClicksRow(probs: Float64Array, clickMatrix: ArrayLike<number>, perms: Perms): number {
  const m = perms.m;
  let total = 0;
  for (let p = 0; p < perms.count; p++) {
    if (probs[p] === 0) continue;
    let v = 0;
    for (let k = 0; k < m; k++) v += clickMatrix[perms.slots[p * m + k] * m + k];
    total += probs[p] * v;
  }
  return total;
}
