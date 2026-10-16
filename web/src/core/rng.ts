// Deterministic random numbers, bit-for-bit the same as betterrankings/rng.py.
// Mulberry32: the state advances by a constant each draw and the output is a
// hash of the state.

const STEP = 0x6d2b79f5;
const TWO_32 = 4294967296;

function hash32(a: number): number {
  let t = Math.imul(a ^ (a >>> 15), a | 1);
  t = (t + Math.imul(t ^ (t >>> 7), t | 61)) ^ t;
  return (t ^ (t >>> 14)) >>> 0;
}

export function fnv1a(text: string): number {
  let h = 0x811c9dc5;
  for (const byte of new TextEncoder().encode(text)) {
    h = Math.imul(h ^ byte, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

export function deriveSeed(seed: number, name: string): number {
  return hash32((((seed >>> 0) ^ fnv1a(name)) + STEP) | 0);
}

export class Rng {
  private a: number;

  constructor(seed: number, name?: string) {
    this.a = name === undefined ? seed >>> 0 : deriveSeed(seed, name);
  }

  next(): number {
    this.a = (this.a + STEP) | 0;
    return hash32(this.a) / TWO_32;
  }

  uniforms(n: number): Float64Array {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.next();
    return out;
  }

  /** Box-Muller, one normal per pair of uniforms. */
  normal(): number {
    const u1 = this.next();
    const u2 = this.next();
    return Math.sqrt(-2 * Math.log(1 - u1)) * Math.cos(2 * Math.PI * u2);
  }

  normals(n: number): Float64Array {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.normal();
    return out;
  }
}
