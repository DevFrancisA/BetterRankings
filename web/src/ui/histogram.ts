// Weight distribution on a log scale, with the clip level and weight 1 marked.

import type { Histogram } from "../protocol";
import { fmtInt } from "./format";
import { linear, niceStep, s, ticks } from "./svg";

const H = 230;
const TOP = 30;
const BOTTOM = 52;
const LEFT = 44;

export function histogramHeight(): number {
  return H;
}

export function weightHistogram(
  h: Histogram,
  clip: number | null,
  nClipped: number,
  width: number,
  title: string,
): SVGSVGElement {
  const W = Math.max(280, width);
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "histogram", role: "img" });
  svg.append(s("title", {}, [title]));
  if (!h.edges.length) {
    svg.append(
      s("text", { x: LEFT, y: H / 2, class: "chart-empty" }, ["Every weight is zero: no logged list matches this policy."]),
    );
    return svg;
  }
  const lo = Math.min(h.edges[0], 0, clip !== null ? Math.log10(clip) : Infinity);
  const hi = Math.max(h.edges[h.edges.length - 1], 0.4, clip !== null ? Math.log10(clip) + 0.2 : -Infinity);
  const x = linear([Math.floor(lo), Math.ceil(hi)], [LEFT, W - 12]);
  const maxCount = Math.max(...h.counts, 1);
  const yStep = niceStep(maxCount, 3);
  const yMax = Math.ceil(maxCount / yStep) * yStep;
  const y = linear([0, yMax], [H - BOTTOM, TOP]);

  const grid = s("g", { "aria-hidden": "true" });
  for (const t of ticks([0, yMax], yStep)) {
    grid.append(s("line", { x1: LEFT, x2: W - 12, y1: y(t), y2: y(t), class: "grid-major" }));
    grid.append(s("text", { x: LEFT - 8, y: y(t) + 4, class: "tick-label", "text-anchor": "end" }, [fmtInt(t)]));
  }
  const [d0, d1] = x.domain;
  for (let dec = d0; dec <= d1; dec++) {
    grid.append(s("line", { x1: x(dec), x2: x(dec), y1: H - BOTTOM, y2: H - BOTTOM + 8, class: "tick-major" }));
    grid.append(
      s("text", { x: x(dec), y: H - BOTTOM + 24, class: "tick-label", "text-anchor": "middle" }, [decadeLabel(dec)]),
    );
    for (let k = 2; k < 10 && dec < d1; k++) {
      const v = dec + Math.log10(k);
      grid.append(s("line", { x1: x(v), x2: x(v), y1: H - BOTTOM, y2: H - BOTTOM + 4, class: "tick-minor" }));
    }
  }
  grid.append(s("line", { x1: LEFT, x2: W - 12, y1: H - BOTTOM, y2: H - BOTTOM, class: "axis-rule" }));
  grid.append(
    s("text", { x: W - 12, y: H - 8, class: "axis-title", "text-anchor": "end" }, [
      W < 520 ? "Weight (log)" : "Importance weight = target ÷ logging probability (log scale)",
    ]),
  );
  svg.append(grid);

  const bars = s("g", { class: "hist-bars" });
  h.counts.forEach((c, i) => {
    if (!c) return;
    const x0 = x(h.edges[i]) + 1;
    const x1 = x(h.edges[i + 1]) - 1;
    const bar = s("rect", { x: x0, y: y(c), width: Math.max(1, x1 - x0), height: H - BOTTOM - y(c), class: "hist-bar" });
    bar.append(
      s("title", {}, [`${fmtInt(c)} queries with weight ${decadeValue(h.edges[i])} to ${decadeValue(h.edges[i + 1])}`]),
    );
    bars.append(bar);
  });
  svg.append(bars);

  const one = x(0);
  svg.append(
    s("line", { x1: one, x2: one, y1: TOP - 6, y2: H - BOTTOM, class: "ref-line" }),
    s("text", { x: one + 5, y: TOP - 10, class: "ref-label" }, ["weight 1"]),
  );
  if (clip !== null) {
    const xc = x(Math.log10(clip));
    const right = xc < W * 0.6;
    svg.append(
      s("line", { x1: xc, x2: xc, y1: TOP - 6, y2: H - BOTTOM, class: "clip-line" }),
      s("text", { x: xc + (right ? 6 : -6), y: TOP + 8, class: "clip-label", "text-anchor": right ? "start" : "end" }, [
        `clip ${clip} · ${fmtInt(nClipped)} cut`,
      ]),
    );
  }
  return svg;
}

function decadeLabel(dec: number): string {
  return decadeValue(dec);
}

function decadeValue(l: number): string {
  const v = Math.pow(10, l);
  if (v >= 1000) return fmtInt(v);
  if (v >= 1) return String(+v.toPrecision(2));
  return String(+v.toPrecision(2));
}
