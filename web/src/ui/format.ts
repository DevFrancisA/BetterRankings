export const fmt = (x: number | null | undefined, digits = 3): string =>
  x === null || x === undefined || !Number.isFinite(x) ? "n/a" : x.toFixed(digits);

export const fmtInt = (x: number): string => Math.round(x).toLocaleString("en-US");

export const fmtPct = (x: number, digits = 0): string =>
  Number.isFinite(x) ? `${(x * 100).toFixed(digits)}%` : "n/a";

/** Signed gap with a proper minus sign, for example "−0.026" or "+0.004". */
export const fmtGap = (x: number, digits = 3): string =>
  !Number.isFinite(x) ? "n/a" : `${x < 0 ? "−" : "+"}${Math.abs(x).toFixed(digits)}`;

export const fmtWeight = (x: number): string => {
  if (!Number.isFinite(x)) return "n/a";
  if (x >= 100) return fmtInt(x);
  if (x >= 10) return x.toFixed(1);
  return x.toFixed(2);
};

export const fmtClip = (c: number | null): string => (c === null ? "No clipping" : `Clip at ${c}`);
