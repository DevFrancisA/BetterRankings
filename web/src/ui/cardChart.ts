// The card chart: one column per row, each bar drawn from a baseline (the true
// value, the bar to beat, or the logged rate) to the estimate. Only the gap is
// shown up front; hovering, focusing or tapping a column reveals its value and
// 95% range. Positions are percentages, so updates glide with CSS transitions.

import { fmt, fmtGap } from "./format";
import { paddedDomain } from "./svg";

export interface CardRow {
  id: string;
  /** Short label under the column. */
  label: string;
  /** Full name, used in the detail panel and for screen readers. */
  name: string;
  value: number;
  lo: number;
  hi: number;
  /** Per-row true value, drawn as a diamond (the ranking view). */
  truth?: number;
  available: boolean;
  /** "main" bars carry the series colour; "muted" bars recede. */
  tone: "main" | "muted";
  /** Optional verdict, e.g. "Worth a live test". */
  note?: string;
  unavailableText?: string;
}

export interface CardSpec {
  rows: CardRow[];
  baseline: { value: number; label: string };
  legend: { main: string; muted?: string; marker?: string };
  /** Accessible name for the whole chart. */
  title: string;
  /** Extra values the axis must cover. */
  domainHint?: number[];
}

interface ColEls {
  col: HTMLButtonElement;
  bar: HTMLSpanElement;
  ci: HTMLSpanElement;
  mark: HTMLSpanElement | null;
  na: HTMLSpanElement;
  label: HTMLSpanElement;
}

function el<K extends keyof HTMLElementTagNameMap>(tag: K, cls = "", text?: string): HTMLElementTagNameMap[K] {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
}

function tickDigits(step: number): number {
  return Math.max(0, -Math.floor(Math.log10(step) + 1e-9));
}

export class CardChart {
  private root: HTMLElement | null = null;
  private plot!: HTMLElement;
  private tickLayer!: HTMLElement;
  private base!: HTMLElement;
  private baseLabel!: HTMLElement;
  private detail!: HTMLElement;
  private cols = new Map<string, ColEls>();
  private spec: CardSpec | null = null;
  private active: string | null = null;

  constructor(private host: HTMLElement) {}

  /** Build from scratch; `enter` grows the bars in. */
  render(spec: CardSpec, enter: boolean): void {
    this.spec = spec;
    this.active = null;
    this.cols.clear();

    const root = el("div", `cc${enter ? " is-entering" : ""}`);
    root.setAttribute("role", "group");
    root.setAttribute("aria-label", spec.title);
    root.style.setProperty("--n", String(spec.rows.length));

    for (const layer of ["cc-glow", "cc-wash"]) {
      const d = el("div", layer);
      d.setAttribute("aria-hidden", "true");
      root.append(d);
    }

    // Top band: the legend pills, swapped for the detail panel while a column is active.
    const top = el("div", "cc-top");
    top.setAttribute("aria-hidden", "true");
    const pills = el("div", "cc-pills");
    const pill = (swatch: string, text: string) => {
      const p = el("span", "cc-pill");
      p.append(el("i", swatch), text);
      return p;
    };
    pills.append(pill("sw-main", spec.legend.main));
    if (spec.legend.muted) pills.append(pill("sw-muted", spec.legend.muted));
    if (spec.legend.marker) pills.append(pill("sw-marker", spec.legend.marker));
    this.detail = el("div", "cc-detail");
    top.append(pills, this.detail);
    root.append(top);

    this.plot = el("div", "cc-plot");
    this.tickLayer = el("div", "cc-ticks");
    this.tickLayer.setAttribute("aria-hidden", "true");
    this.base = el("div", "cc-base");
    this.base.setAttribute("aria-hidden", "true");
    this.baseLabel = el("span", "cc-base-label");
    this.base.append(this.baseLabel);
    const colsEl = el("div", "cc-cols");
    const labels = el("div", "cc-labels");
    labels.setAttribute("aria-hidden", "true");

    spec.rows.forEach((r, i) => {
      const col = el("button", "cc-col");
      col.type = "button";
      col.style.setProperty("--i", String(i));
      const ci = el("span", "cc-ci");
      const bar = el("span", "cc-bar");
      const na = el("span", "cc-na", "n/a");
      let mark: HTMLSpanElement | null = null;
      col.append(ci, bar);
      if (r.truth !== undefined) {
        mark = el("span", "cc-mark");
        col.append(mark);
      }
      col.append(na);
      const label = el("span", "cc-label", r.label);
      labels.append(label);
      colsEl.append(col);
      this.cols.set(r.id, { col, bar, ci, mark, na, label });

      col.addEventListener("pointerenter", () => this.setActive(r.id));
      col.addEventListener("focus", () => this.setActive(r.id));
      col.addEventListener("click", () => this.setActive(r.id));
      col.addEventListener("blur", () => {
        requestAnimationFrame(() => {
          if (!this.root?.contains(document.activeElement)) this.setActive(null);
        });
      });
    });
    root.addEventListener("pointerleave", () => {
      if (!root.contains(document.activeElement)) this.setActive(null);
    });

    this.plot.append(this.tickLayer, colsEl, this.base);
    root.append(this.plot, labels);
    this.root?.remove();
    this.host.replaceChildren(root);
    this.root = root;
    this.place(spec);
  }

  /** Move to new values; rebuilds only when the set of rows changes. */
  update(spec: CardSpec): void {
    const same =
      this.root && this.spec && spec.rows.map((r) => r.id).join("|") === this.spec.rows.map((r) => r.id).join("|");
    if (!same) return this.render(spec, false);
    this.spec = spec;
    this.place(spec);
  }

  private place(spec: CardSpec): void {
    const base = spec.baseline.value;
    // Fit the visible marks only. Ranges appear one at a time and stop at the edge,
    // so a single wide interval does not flatten every bar.
    const deltas = [0, ...(spec.domainHint ?? []).map((v) => v - base)];
    for (const r of spec.rows) {
      if (!r.available) continue;
      deltas.push(r.value - base);
      if (r.truth !== undefined) deltas.push(r.truth - base);
    }
    const { domain, step } = paddedDomain(deltas, 4);
    const [d0, d1] = domain;
    const y = (d: number) => ((d1 - d) / (d1 - d0)) * 100;

    // Two quiet gridlines, at the top and bottom of the scale, in units of
    // "difference from the baseline". The baseline itself is the third line.
    const digits = tickDigits(step);
    this.tickLayer.replaceChildren(
      ...[d1, d0]
        .filter((t) => Math.abs(t) > step / 2)
        .map((t) => {
          const line = el("span", "cc-tick");
          line.style.top = `${y(t)}%`;
          line.append(el("span", "cc-tick-label", fmtGap(t, digits)));
          return line;
        }),
    );

    this.base.style.top = `${y(0)}%`;
    this.baseLabel.textContent = `${spec.baseline.label} ${fmt(base)}`;
    this.baseLabel.classList.toggle("is-low", y(0) < 18);

    for (const r of spec.rows) {
      const c = this.cols.get(r.id);
      if (!c) continue;
      c.col.classList.toggle("tone-muted", r.tone === "muted");
      c.col.classList.toggle("is-na", !r.available);
      c.col.setAttribute("aria-label", this.describe(r, spec));
      c.na.style.top = `${y(0)}%`;
      if (r.available) {
        const v = r.value - base;
        const top = y(Math.max(v, 0));
        const bottom = y(Math.min(v, 0));
        c.bar.style.top = `${top}%`;
        c.bar.style.height = `${bottom - top}%`;
        c.bar.classList.toggle("down", v < 0);
        const hasCi = Number.isFinite(r.lo) && Number.isFinite(r.hi);
        c.ci.hidden = !hasCi;
        if (hasCi) {
          const raw0 = y(r.hi - base);
          const raw1 = y(r.lo - base);
          const t0 = Math.max(0, raw0);
          const t1 = Math.min(100, raw1);
          c.ci.style.top = `${t0}%`;
          c.ci.style.height = `${Math.max(0, t1 - t0)}%`;
          c.ci.classList.toggle("open-top", raw0 < 0);
          c.ci.classList.toggle("open-bottom", raw1 > 100);
        }
      }
      if (c.mark && r.truth !== undefined) c.mark.style.top = `${y(r.truth - base)}%`;
    }
    if (this.active) this.setActive(this.active);
  }

  private describe(r: CardRow, spec: CardSpec): string {
    if (!r.available) return `${r.name}: ${r.unavailableText ?? "no estimate"}`;
    const parts = [
      `${r.name}: ${fmt(r.value)}`,
      `${fmtGap(r.value - spec.baseline.value)} vs ${spec.baseline.label.toLowerCase()}`,
    ];
    if (Number.isFinite(r.lo) && Number.isFinite(r.hi)) parts.push(`95% range ${fmt(r.lo)} to ${fmt(r.hi)}`);
    if (r.truth !== undefined) parts.push(`true value ${fmt(r.truth)}`);
    if (r.note) parts.push(r.note);
    return parts.join(", ");
  }

  private setActive(id: string | null): void {
    if (!this.root || !this.spec) return;
    this.active = id;
    const r = id ? this.spec.rows.find((row) => row.id === id) : undefined;
    this.root.classList.toggle("is-active", !!r);
    for (const [key, c] of this.cols) {
      const on = key === id;
      c.col.classList.toggle("is-on", on);
      c.label.classList.toggle("is-on", on);
    }
    if (!r) return;

    const head = el("p", "cc-detail-head");
    head.append(el("i", r.tone === "muted" ? "sw-muted" : "sw-main"), r.name);
    const body = el("p", "cc-detail-body");
    if (r.available) {
      const b = this.spec.baseline;
      body.append(
        el("span", "cc-num", fmt(r.value)),
        ` ${fmtGap(r.value - b.value)} vs ${b.label.toLowerCase()}`,
      );
      if (Number.isFinite(r.lo) && Number.isFinite(r.hi))
        body.append(el("span", "cc-sep", "·"), `range ${fmt(r.lo)}–${fmt(r.hi)}`);
      if (r.truth !== undefined) body.append(el("span", "cc-sep", "·"), `true ${fmt(r.truth)}`);
      if (r.note) body.append(el("span", "cc-sep", "·"), r.note);
    } else {
      body.textContent = r.unavailableText ?? "No estimate";
    }
    this.detail.replaceChildren(head, body);
  }
}
