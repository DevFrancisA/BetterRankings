// Small helpers for hand-built SVG charts.

export const SVG_NS = "http://www.w3.org/2000/svg";

type Attrs = Record<string, string | number | undefined>;

export function s<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  children: (Node | string)[] = [],
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined) continue;
    if (k === "style") node.setAttribute("style", String(v));
    else node.setAttribute(k, String(v));
  }
  for (const c of children) node.append(typeof c === "string" ? document.createTextNode(c) : c);
  return node;
}

export interface Scale {
  (v: number): number;
  domain: [number, number];
  range: [number, number];
}

export function linear(domain: [number, number], range: [number, number]): Scale {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const k = d1 === d0 ? 0 : (r1 - r0) / (d1 - d0);
  const f = ((v: number) => r0 + (v - d0) * k) as Scale;
  f.domain = domain;
  f.range = range;
  return f;
}

/** A "nice" tick step near span / target, from the 1-2-5 sequence. */
export function niceStep(span: number, target: number): number {
  const raw = span / Math.max(1, target);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const r = raw / mag;
  return (r < 1.5 ? 1 : r < 3.5 ? 2 : r < 7.5 ? 5 : 10) * mag;
}

export function ticks(domain: [number, number], step: number): number[] {
  const out: number[] = [];
  const start = Math.ceil(domain[0] / step - 1e-9) * step;
  for (let v = start; v <= domain[1] + step * 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}

/** Domain covering the values with some padding, snapped outward to the tick step. */
export function paddedDomain(values: number[], targetTicks: number): { domain: [number, number]; step: number } {
  const finite = values.filter(Number.isFinite);
  let lo = Math.min(...finite);
  let hi = Math.max(...finite);
  if (!finite.length) {
    lo = 0;
    hi = 1;
  }
  if (hi - lo < 1e-6) {
    lo -= 0.05;
    hi += 0.05;
  }
  const pad = (hi - lo) * 0.06;
  lo -= pad;
  hi += pad;
  const step = niceStep(hi - lo, targetTicks);
  return { domain: [Math.floor(lo / step) * step, Math.ceil(hi / step) * step], step };
}

export function diamond(size: number): string {
  return `M0,${-size} L${size},0 L0,${size} L${-size},0 Z`;
}

/** Plays entry animations only when the user has not asked for reduced motion. */
export function prefersReducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}
