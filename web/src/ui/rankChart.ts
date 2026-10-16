// Rank agreement: the true order on the left, one estimator's order on the
// right, lines joining each policy. Crossing lines are ranking mistakes.

import { diamond, s } from "./svg";

export interface RankItem {
  id: string;
  label: string;
  truth: number;
  estimate: number;
}

const ROW = 30;
const TOP = 34;

export function rankChart(items: RankItem[], estimatorLabel: string, width: number): SVGSVGElement {
  const byTruth = [...items].sort((a, b) => b.truth - a.truth);
  const byEst = [...items].sort((a, b) => {
    const ea = Number.isFinite(a.estimate) ? a.estimate : -Infinity;
    const eb = Number.isFinite(b.estimate) ? b.estimate : -Infinity;
    return eb - ea;
  });
  const W = Math.max(270, width);
  const H = TOP + items.length * ROW + 6;
  const labelW = Math.min(118, (W - 70) / 2);
  const xl = labelW + 8;
  const xr = W - labelW - 8;
  const y = (rank: number) => TOP + rank * ROW + ROW / 2 - 4;
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "rank-chart", "aria-hidden": "true" });

  svg.append(
    s("text", { x: xl, y: 14, class: "rank-head", "text-anchor": "end" }, ["True order"]),
    s("text", { x: xr, y: 14, class: "rank-head", "text-anchor": "start" }, [estimatorLabel]),
    s("line", { x1: 0, x2: W, y1: 22, y2: 22, class: "axis-rule" }),
  );
  byTruth.forEach((item, rt) => {
    const re = byEst.findIndex((e) => e.id === item.id);
    const moved = re !== rt;
    const missing = !Number.isFinite(item.estimate);
    svg.append(
      s("line", {
        x1: xl + 8,
        x2: xr - 8,
        y1: y(rt),
        y2: y(re),
        class: `rank-link${moved ? " rank-link-moved" : ""}${missing ? " rank-link-missing" : ""}`,
      }),
    );
  });
  byTruth.forEach((item, rt) => {
    svg.append(
      s("text", { x: xl - 10, y: y(rt) + 4, class: "rank-label", "text-anchor": "end" }, [item.label]),
      s("path", { d: diamond(5), transform: `translate(${xl},${y(rt)})`, class: "rank-truth" }),
    );
  });
  byEst.forEach((item, re) => {
    svg.append(
      s("circle", { cx: xr, cy: y(re), r: 4.5, class: Number.isFinite(item.estimate) ? "rank-est" : "rank-est-missing" }),
      s("text", { x: xr + 10, y: y(re) + 4, class: "rank-label" }, [item.label]),
    );
  });
  return svg;
}
