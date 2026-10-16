// The forest plot: one shared horizontal axis, a full-height truth line, and a
// row per estimate with a dot and a confidence-interval whisker. Marks are
// positioned with CSS transforms so updates slide instead of jumping.

import { fmt } from "./format";
import { diamond, linear, paddedDomain, s, ticks, type Scale } from "./svg";

export interface ForestRow {
  id: string;
  label: string;
  /** Used instead of label when the chart is narrow. */
  shortLabel?: string;
  /** Right-aligned text on the label line, e.g. "0.828  −0.026". */
  value: string;
  /** Optional second part of the value text, styled quieter. */
  note?: string;
  noteShort?: string;
  noteKind?: "good" | "bad" | "plain";
  estimate: number;
  lo: number;
  hi: number;
  /** Per-row true value (policy ranking view). */
  truth?: number;
  available: boolean;
  unavailableText?: string;
  emphasis?: boolean;
}

export interface ForestSpec {
  rows: ForestRow[];
  truthLine?: { value: number; label: string };
  reference?: { value: number; label: string };
  axisLabel: string;
  /** Extra values the axis must cover (other clip levels, other estimators). */
  domainHint?: number[];
  title: string;
  description: string;
}

const TOP = 46;
const ROW = 58;
const MARK_Y = 40;
const AXIS_BAND = 76;
const NARROW = 520;

interface RowEls {
  whisker: SVGRectElement;
  capLo: SVGRectElement;
  capHi: SVGRectElement;
  dot: SVGCircleElement;
  value: SVGTSpanElement;
  note: SVGTSpanElement;
  truth?: SVGPathElement;
  missing: SVGTextElement;
}

export class ForestPlot {
  private svg: SVGSVGElement | null = null;
  private rows = new Map<string, RowEls>();
  private x: Scale = linear([0, 1], [0, 1]);
  private width = 0;
  private spec: ForestSpec | null = null;

  constructor(private host: HTMLElement) {
    new ResizeObserver(() => {
      const w = Math.round(this.host.clientWidth);
      if (this.spec && w > 0 && w !== this.width) this.render(this.spec, false);
    }).observe(host);
  }

  static height(nRows: number): number {
    return TOP + nRows * ROW + AXIS_BAND;
  }

  /** Draw from scratch; `enter` plays the arrival sequence. */
  render(spec: ForestSpec, enter: boolean): void {
    this.spec = spec;
    this.width = Math.max(260, Math.round(this.host.clientWidth));
    const W = this.width;
    const H = ForestPlot.height(spec.rows.length);
    const values: number[] = [...(spec.domainHint ?? [])];
    for (const r of spec.rows) values.push(r.estimate, r.lo, r.hi, r.truth ?? NaN);
    if (spec.truthLine) values.push(spec.truthLine.value);
    if (spec.reference) values.push(spec.reference.value);
    const { domain, step } = paddedDomain(values, W < 480 ? 4 : 7);
    this.x = linear(domain, [10, W - 10]);
    const x = this.x;
    const bottom = TOP + spec.rows.length * ROW;
    const rowsEls = new Map<string, RowEls>();

    const titleId = `t-${Math.random().toString(36).slice(2)}`;
    const svg = s("svg", {
      class: `forest${enter ? " is-entering" : ""}`,
      viewBox: `0 0 ${W} ${H}`,
      width: W,
      height: H,
      role: "img",
      "aria-labelledby": `${titleId} ${titleId}-d`,
    });
    svg.append(s("title", { id: titleId }, [spec.title]), s("desc", { id: `${titleId}-d` }, [spec.description]));

    // Grid and ticks
    const grid = s("g", { class: "grid", "aria-hidden": "true" });
    const major = ticks(domain, step);
    const minor = ticks(domain, step / 5);
    for (const t of major) grid.append(s("line", { x1: x(t), x2: x(t), y1: TOP - 10, y2: bottom + 6, class: "grid-major" }));
    for (const t of minor) {
      grid.append(s("line", { x1: x(t), x2: x(t), y1: TOP - 14, y2: TOP - 10, class: "tick-minor" }));
      grid.append(s("line", { x1: x(t), x2: x(t), y1: bottom + 6, y2: bottom + 10, class: "tick-minor" }));
    }
    for (const t of major) {
      grid.append(s("line", { x1: x(t), x2: x(t), y1: bottom + 6, y2: bottom + 14, class: "tick-major" }));
      grid.append(s("text", { x: x(t), y: bottom + 30, class: "tick-label", "text-anchor": "middle" }, [fmtTick(t, step)]));
    }
    grid.append(s("line", { x1: 0, x2: W, y1: bottom + 6, y2: bottom + 6, class: "axis-rule" }));
    grid.append(s("line", { x1: 0, x2: W, y1: TOP - 10, y2: TOP - 10, class: "axis-rule axis-rule-top" }));
    grid.append(s("text", { x: 0, y: bottom + 68, class: "axis-title" }, [`${spec.axisLabel} →`]));
    svg.append(grid);

    // Reference line (bar to beat / logged value), quiet ink.
    if (spec.reference && Number.isFinite(spec.reference.value)) {
      const xr = x(spec.reference.value);
      const g = s("g", { class: "reference" });
      g.append(s("line", { x1: xr, x2: xr, y1: TOP - 10, y2: bottom + 6, class: "ref-line" }));
      const anchor = xr > W * 0.6 ? "end" : "start";
      g.append(
        s("text", { x: xr + (anchor === "end" ? -6 : 6), y: bottom + 48, class: "ref-label", "text-anchor": anchor }, [
          spec.reference.label,
        ]),
      );
      svg.append(g);
    }

    // Truth line, drawn under the rows so their labels stay readable over it.
    if (spec.truthLine && Number.isFinite(spec.truthLine.value)) {
      const xt = x(spec.truthLine.value);
      const len = bottom + 6 - (TOP - 18);
      const g = s("g", { class: "truth" });
      g.append(
        s("line", {
          class: "truth-line",
          x1: xt,
          x2: xt,
          y1: TOP - 18,
          y2: bottom + 6,
          style: `--len:${len}`,
        }),
      );
      g.append(s("path", { class: "truth-cap", d: diamond(7), transform: `translate(${xt},${TOP - 18})` }));
      const right = xt < W * 0.55;
      g.append(
        s(
          "text",
          { class: "truth-label", x: xt + (right ? 12 : -12), y: TOP - 22, "text-anchor": right ? "start" : "end" },
          [
            s("tspan", { class: "truth-label-name" }, [spec.truthLine.label]),
            s("tspan", { class: "truth-label-value", dx: 6 }, [fmt(spec.truthLine.value)]),
          ],
        ),
      );
      svg.append(g);
    }

    // Rows
    spec.rows.forEach((r, i) => {
      const g = s("g", { class: `row${r.emphasis ? " row-emphasis" : ""}`, transform: `translate(0,${TOP + i * ROW})` });
      const land = s("g", { class: "land", style: `--i:${i}` });
      land.append(s("rect", { class: "row-band", x: 0, y: 2, width: W, height: ROW - 4 }));
      land.append(s("line", { class: "row-rule", x1: 0, x2: W, y1: MARK_Y, y2: MARK_Y }));
      land.append(s("text", { class: "row-label", x: 0, y: 18 }, [W < NARROW ? (r.shortLabel ?? r.label) : r.label]));
      const value = s("tspan", { class: "row-value-main" });
      const note = s("tspan", { class: "row-note", dx: 8 });
      land.append(s("text", { class: "row-value", x: W, y: 18, "text-anchor": "end" }, [value, note]));
      const whisker = s("rect", { class: "whisker mark-el", x: 0, y: MARK_Y - 1, width: 1, height: 2 });
      const capLo = s("rect", { class: "cap mark-el", x: -0.75, y: MARK_Y - 6, width: 1.5, height: 12 });
      const capHi = s("rect", { class: "cap mark-el", x: -0.75, y: MARK_Y - 6, width: 1.5, height: 12 });
      let truth: SVGPathElement | undefined;
      if (r.truth !== undefined) {
        truth = s("path", { class: "row-truth mark-el", d: diamond(6.5) });
        truth.style.transform = `translate(${x(r.truth)}px, ${MARK_Y}px)`;
      }
      const dot = s("circle", { class: "dot mark-el", cx: 0, cy: MARK_Y, r: 6 });
      const missing = s("text", { class: "row-missing", x: 0, y: MARK_Y + 5 });
      dot.append(s("title"));
      land.append(whisker, capLo, capHi);
      if (truth) land.append(truth);
      land.append(dot, missing);
      g.append(land);
      svg.append(g);
      rowsEls.set(r.id, { whisker, capLo, capHi, dot, value, note, truth, missing });
    });

    this.svg?.remove();
    this.host.replaceChildren(svg);
    this.svg = svg;
    this.rows = rowsEls;
    this.place(spec);
  }

  /** Move marks to new values; redraws only when the rows or axis must change. */
  update(spec: ForestSpec): void {
    if (!this.svg || !this.spec) return this.render(spec, false);
    const values: number[] = [];
    for (const r of spec.rows) values.push(r.lo, r.hi, r.estimate);
    const [d0, d1] = this.x.domain;
    const fits = values.filter(Number.isFinite).every((v) => v >= d0 && v <= d1);
    const sameRows = spec.rows.map((r) => r.id).join("|") === this.spec.rows.map((r) => r.id).join("|");
    if (!fits || !sameRows) {
      this.render(spec, false);
      return;
    }
    this.spec = spec;
    this.place(spec);
  }

  private place(spec: ForestSpec): void {
    const x = this.x;
    for (const r of spec.rows) {
      const els = this.rows.get(r.id);
      if (!els) continue;
      els.value.textContent = r.value;
      els.note.textContent = (this.width < NARROW ? (r.noteShort ?? r.note) : r.note) ?? "";
      els.note.setAttribute("class", `row-note${r.noteKind ? ` note-${r.noteKind}` : ""}`);
      const show = r.available && Number.isFinite(r.estimate);
      const hasCi = show && Number.isFinite(r.lo) && Number.isFinite(r.hi);
      const xl = hasCi ? x(r.lo) : 0;
      const xh = hasCi ? x(r.hi) : 0;
      els.whisker.style.transform = `translate(${xl}px, 0) scaleX(${Math.max(0.001, xh - xl)})`;
      els.capLo.style.transform = `translateX(${xl}px)`;
      els.capHi.style.transform = `translateX(${xh}px)`;
      for (const e of [els.whisker, els.capLo, els.capHi]) e.style.opacity = hasCi ? "1" : "0";
      els.dot.style.transform = `translateX(${show ? x(r.estimate) : 0}px)`;
      els.dot.style.opacity = show ? "1" : "0";
      els.dot.querySelector("title")!.textContent = show
        ? `${r.label}: ${fmt(r.estimate)}, 95% interval ${fmt(r.lo)} to ${fmt(r.hi)}`
        : r.label;
      if (els.truth && r.truth !== undefined) els.truth.style.transform = `translate(${x(r.truth)}px, ${MARK_Y}px)`;
      els.missing.textContent = show ? "" : (r.unavailableText ?? "Not available");
    }
  }
}

function fmtTick(v: number, step: number): string {
  const digits = Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
  return v.toFixed(digits);
}
