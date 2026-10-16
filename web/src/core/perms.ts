// Every ranking of m slots, in lexicographic order (same as itertools.permutations).

export const MAX_SLOTS = 6;

export interface Perms {
  m: number;
  count: number;
  /** Flat (count * m): slot shown at each position, row by row. */
  slots: Int32Array;
}

const cache = new Map<number, Perms>();

export function permutations(m: number): Perms {
  const hit = cache.get(m);
  if (hit) return hit;
  if (m < 1 || m > MAX_SLOTS) throw new Error(`lists must have between 1 and ${MAX_SLOTS} items, got ${m}`);
  const rows: number[][] = [];
  const used = new Array<boolean>(m).fill(false);
  const cur: number[] = [];
  const walk = () => {
    if (cur.length === m) {
      rows.push(cur.slice());
      return;
    }
    for (let j = 0; j < m; j++) {
      if (used[j]) continue;
      used[j] = true;
      cur.push(j);
      walk();
      cur.pop();
      used[j] = false;
    }
  };
  walk();
  const slots = new Int32Array(rows.length * m);
  rows.forEach((r, p) => slots.set(r, p * m));
  const out = { m, count: rows.length, slots };
  cache.set(m, out);
  return out;
}

function factorial(k: number): number {
  let f = 1;
  for (let i = 2; i <= k; i++) f *= i;
  return f;
}

/** Lexicographic index of one ranking (Lehmer code). */
export function permIndex(order: ArrayLike<number>): number {
  const m = order.length;
  let idx = 0;
  for (let k = 0; k < m; k++) {
    let smaller = 0;
    for (let j = k + 1; j < m; j++) if (order[j] < order[k]) smaller++;
    idx += smaller * factorial(m - 1 - k);
  }
  return idx;
}
