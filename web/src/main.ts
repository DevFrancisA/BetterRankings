import "./styles.css";

import { ESTIMATOR_NAMES, ESTIMATORS, type Estimate, type EstimatorId } from "./core/estimators";
import { DEFAULTS, discordantPairs, TARGET_COLUMN, type TargetId } from "./core/evaluate";
import { POLICIES, POLICY_BY_ID, POLICY_IDS, type PolicyId } from "./core/policies";
import { CLIP_LEVELS, type EvalView, type PolicyView, type Request, type Response, type RunParams } from "./protocol";
import { ForestPlot, type ForestRow, type ForestSpec } from "./ui/forest";
import { fmt, fmtClip, fmtGap, fmtInt, fmtPct, fmtWeight } from "./ui/format";
import { weightHistogram } from "./ui/histogram";
import { rankChart } from "./ui/rankChart";

// DOM helpers ------------------------------------------------------------------

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string> = {},
  children: (Node | string)[] = [],
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  el.append(...children);
  return el;
}

function setText(id: string, text: string) {
  $(id).textContent = text;
}

function download(name: string, text: string, type = "text/csv") {
  const url = URL.createObjectURL(new Blob([text], { type }));
  const a = h("a", { href: url, download: name });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function csvCell(v: string | number | null | undefined): string {
  if (v === null || v === undefined || (typeof v === "number" && !Number.isFinite(v))) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}

// Worker ---------------------------------------------------------------------

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
let nextId = 1;
const pending = new Map<
  number,
  { resolve: (r: Response) => void; reject: (e: Error) => void; progress?: (stage: string, f: number) => void }
>();

type RequestBody = Request extends infer R ? (R extends { id: number } ? Omit<R, "id"> : never) : never;

function call(body: RequestBody, progress?: (stage: string, f: number) => void): Promise<Response> {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject, progress });
    worker.postMessage({ ...body, id } as Request);
  });
}

worker.onmessage = (e: MessageEvent<Response>) => {
  const msg = e.data;
  const p = pending.get(msg.id);
  if (!p) return;
  if (msg.type === "progress") return p.progress?.(msg.stage, msg.fraction);
  pending.delete(msg.id);
  if (msg.type === "error") p.reject(new Error(msg.message));
  else p.resolve(msg);
};

worker.onerror = () => {
  for (const p of pending.values()) p.reject(new Error("The computation worker stopped. Reload the page and try again."));
  pending.clear();
};

// State ----------------------------------------------------------------------

const DEFAULT_CLIP_INDEX = CLIP_LEVELS.indexOf(DEFAULTS.clip);
const state = {
  sim: null as EvalView | null,
  enter: false,
  policy: "relevance" as PolicyId,
  clipIdx: DEFAULT_CLIP_INDEX,
  rankEst: "doubly_robust" as EstimatorId,
  running: false,
  tab: "results",
  dirty: new Set<string>(["results", "ranking", "diagnostics"]),
  upload: {
    text: null as string | null,
    fileName: "",
    hasTarget: false,
    view: null as EvalView | null,
    clipIdx: DEFAULT_CLIP_INDEX,
    enter: false,
  },
};

const plots = {
  results: new ForestPlot($("forest")),
  ranking: new ForestPlot($("ranking-forest")),
  upload: new ForestPlot($("upload-forest")),
};

// Controls -------------------------------------------------------------------

/** Range inputs paint their track up to the thumb from --fill. */
function syncFill(input: HTMLInputElement) {
  const min = Number(input.min || 0);
  const max = Number(input.max || 100);
  input.style.setProperty("--fill", `${((Number(input.value) - min) / (max - min)) * 100}%`);
}
for (const r of document.querySelectorAll<HTMLInputElement>('input[type="range"]')) {
  syncFill(r);
  r.addEventListener("input", () => syncFill(r));
}

const form = $<HTMLFormElement>("controls");
const exploration = $<HTMLInputElement>("exploration");
const queries = $<HTMLSelectElement>("queries");
const seed = $<HTMLInputElement>("seed");
const policySelect = $<HTMLSelectElement>("policy");
const clip = $<HTMLInputElement>("clip");

for (const p of POLICIES) policySelect.append(h("option", { value: p.id }, [p.name]));
policySelect.value = state.policy;

function readParams(): RunParams {
  const s = Math.floor(Number(seed.value));
  return {
    nQueries: Number(queries.value),
    exploration: Number(exploration.value),
    seed: Number.isFinite(s) ? Math.min(4294967295, Math.max(0, s)) : DEFAULTS.seed,
    clip: CLIP_LEVELS[state.clipIdx],
    nBoot: DEFAULTS.nBoot,
  };
}

function updateStale() {
  const v = state.sim?.params;
  const p = readParams();
  const stale = !!v && (v.nQueries !== p.nQueries || v.exploration !== p.exploration || v.seed !== p.seed);
  $("results-stale").hidden = !stale;
}

function updatePolicyHint() {
  setText("policy-hint", POLICY_BY_ID[state.policy].description);
}

exploration.addEventListener("input", () => {
  setText("exploration-out", Number(exploration.value).toFixed(2));
  updateStale();
});
queries.addEventListener("change", updateStale);
seed.addEventListener("input", updateStale);
$("new-seed").addEventListener("click", () => {
  seed.value = String(Math.floor(Math.random() * 100000));
  updateStale();
});
policySelect.addEventListener("change", () => {
  state.policy = policySelect.value as PolicyId;
  updatePolicyHint();
  markDirty("results", "diagnostics");
});

form.addEventListener("submit", (e) => {
  e.preventDefault();
  void runEvaluation(true);
});

async function runEvaluation(fromUser = false) {
  if (state.running) return;
  state.running = true;
  const params = readParams();
  seed.value = String(params.seed);
  const button = $<HTMLButtonElement>("run");
  const bar = $<HTMLProgressElement>("run-progress");
  button.disabled = true;
  button.textContent = "Running…";
  bar.hidden = false;
  bar.value = 0;
  $("run-message").classList.remove("error");
  setText("run-message", "Simulating logged traffic…");
  document.body.setAttribute("aria-busy", "true");
  $("forest").classList.add("is-busy");
  const t0 = performance.now();
  try {
    const res = await call({ type: "run", params }, (stage, f) => {
      bar.value = f;
      setText("run-message", `${stage}…`);
    });
    if (res.type !== "result") throw new Error("Unexpected reply from the worker.");
    state.sim = res.view;
    state.enter = true;
    setText(
      "run-message",
      `Done. ${fmtInt(params.nQueries)} logged queries and every candidate evaluated in ${((performance.now() - t0) / 1000).toFixed(1)} s.`,
    );
    $<HTMLButtonElement>("download-results").disabled = false;
    $<HTMLButtonElement>("download-log").disabled = false;
    markDirty("results", "ranking", "diagnostics");
    updateStale();
    const app = $("main");
    if (fromUser && app.getBoundingClientRect().top > window.innerHeight * 0.6) {
      app.scrollIntoView({ behavior: matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
    }
  } catch (err) {
    $("run-message").classList.add("error");
    setText("run-message", `${(err as Error).message} Check the settings and press Run evaluation again.`);
    if (!state.sim) setText("results-summary", "The run did not finish, so there is nothing to show yet.");
  } finally {
    state.running = false;
    button.disabled = false;
    button.textContent = "Run evaluation";
    bar.hidden = true;
    document.body.removeAttribute("aria-busy");
    $("forest").classList.remove("is-busy");
  }
}

// Clip sliders: one request in flight at a time, the latest value wins.

function clipControl(
  input: HTMLInputElement,
  outputId: string,
  session: "sim" | "upload",
  get: () => EvalView | null,
  onDone: (policies: EvalView["policies"], c: number | null) => void,
) {
  let inFlight = false;
  let queued: number | null = null;
  const send = async (idx: number) => {
    inFlight = true;
    try {
      const res = await call({ type: "reclip", session, clip: CLIP_LEVELS[idx] });
      if (res.type === "reclipped") onDone(res.policies, res.clip);
    } catch {
      /* nothing to re-clip yet */
    } finally {
      inFlight = false;
      if (queued !== null) {
        const q = queued;
        queued = null;
        void send(q);
      }
    }
  };
  input.addEventListener("input", () => {
    const idx = Number(input.value);
    setText(outputId, fmtClip(CLIP_LEVELS[idx]));
    if (!get()) return;
    if (inFlight) queued = idx;
    else void send(idx);
  });
}

clip.value = String(state.clipIdx);
syncFill(clip);
clipControl(
  clip,
  "clip-out",
  "sim",
  () => state.sim,
  (policies, c) => {
    if (!state.sim) return;
    state.sim.policies = policies;
    state.sim.clip = c;
    state.clipIdx = Number(clip.value);
    renderResults(true);
    markDirty("ranking", "diagnostics");
  },
);

// Tabs -----------------------------------------------------------------------

const tabs = Array.from(document.querySelectorAll<HTMLButtonElement>('[role="tab"]'));

function selectTab(tab: HTMLButtonElement, focus = false) {
  for (const t of tabs) {
    const on = t === tab;
    t.setAttribute("aria-selected", String(on));
    t.tabIndex = on ? 0 : -1;
    $(t.getAttribute("aria-controls")!).hidden = !on;
  }
  state.tab = tab.id.replace("tab-", "");
  if (focus) tab.focus();
  renderVisible();
}

tabs.forEach((t, i) => {
  t.addEventListener("click", () => selectTab(t));
  t.addEventListener("keydown", (e) => {
    let j = -1;
    if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
    else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
    else if (e.key === "Home") j = 0;
    else if (e.key === "End") j = tabs.length - 1;
    if (j >= 0) {
      e.preventDefault();
      selectTab(tabs[j], true);
    }
  });
});

for (const link of document.querySelectorAll<HTMLAnchorElement>("a[data-tab]")) {
  link.addEventListener("click", () => {
    const tab = $<HTMLButtonElement>(`tab-${link.dataset.tab}`);
    if (tab) selectTab(tab);
  });
}

function markDirty(...names: string[]) {
  for (const n of names) state.dirty.add(n);
  renderVisible();
}

function renderVisible() {
  const t = state.tab;
  if (!state.dirty.has(t)) return;
  state.dirty.delete(t);
  if (t === "results") renderResults(false);
  if (t === "ranking") renderRanking();
  if (t === "diagnostics") renderDiagnostics();
}

// Shared pieces -------------------------------------------------------------------

function estimatorRows(view: PolicyView, truth: number | null): ForestRow[] {
  return ESTIMATORS.map((est) => {
    const e = view.estimates[est];
    const ok = e.available && Number.isFinite(e.estimate);
    let note = "";
    let noteShort = "";
    if (ok && truth !== null) {
      noteShort = fmtGap(e.estimate - truth);
      note = `${noteShort} vs truth`;
    } else if (ok && Number.isFinite(e.ci_low)) note = noteShort = `${fmt(e.ci_low)} to ${fmt(e.ci_high)}`;
    return {
      id: est,
      label: ESTIMATOR_NAMES[est],
      shortLabel: SHORT_ESTIMATOR[est],
      value: ok ? fmt(e.estimate) : "n/a",
      note,
      noteShort,
      estimate: e.estimate,
      lo: e.ci_low,
      hi: e.ci_high,
      available: ok,
      unavailableText: e.available ? "No logged list matches this policy" : "Needs a built-in policy",
    };
  });
}

const SHORT_ESTIMATOR: Record<EstimatorId, string> = {
  naive: "Naive",
  ips: "IPS",
  snips: "Self-normalised IPS",
  clipped_ips: "Clipped IPS",
  reward_model: "Reward model",
  doubly_robust: "Doubly robust",
};

function estimateTable(table: HTMLTableElement, view: PolicyView, truth: number | null) {
  const head = h("tr", {}, [
    h("th", { scope: "col" }, ["Estimator"]),
    h("th", { scope: "col", class: "num" }, ["Estimate"]),
    h("th", { scope: "col", class: "num" }, ["95% interval"]),
    ...(truth !== null ? [h("th", { scope: "col", class: "num" }, ["Gap to truth"])] : []),
    h("th", { scope: "col", class: "num" }, ["Effective sample size"]),
  ]);
  const body = ESTIMATORS.map((est) => {
    const e = view.estimates[est];
    return h("tr", {}, [
      h("th", { scope: "row" }, [ESTIMATOR_NAMES[est]]),
      h("td", { class: "num" }, [e.available ? fmt(e.estimate) : "needs a built-in policy"]),
      h("td", { class: "num" }, [e.available ? `${fmt(e.ci_low)} to ${fmt(e.ci_high)}` : "n/a"]),
      ...(truth !== null ? [h("td", { class: "num" }, [e.available ? fmtGap(e.estimate - truth) : "n/a"])] : []),
      h("td", { class: "num" }, [e.ess === null ? "all rows" : fmtInt(e.ess)]),
    ]);
  });
  table.replaceChildren(h("thead", {}, [head]), h("tbody", {}, body));
}

function notices(host: HTMLElement, diag: PolicyView["diagnostics"], calm?: string) {
  host.replaceChildren();
  for (const w of diag.warnings) {
    host.append(
      h("div", { class: "notice", role: "note" }, [
        warnIcon(),
        h("p", {}, [h("strong", {}, [w.level === "warn" ? "Warning. " : "Note. "]), w.message]),
      ]),
    );
  }
  if (!diag.warnings.length && calm) host.append(h("div", { class: "notice notice-calm" }, [okIcon(), h("p", {}, [calm])]));
}

function warnIcon(): HTMLElement {
  const span = h("span", { class: "notice-icon", "aria-hidden": "true" });
  span.innerHTML =
    '<svg width="18" height="16" viewBox="0 0 18 16"><path d="M9 1 L17 15 H1 Z" fill="none" stroke="#b98900" stroke-width="1.6" stroke-linejoin="round"/><path d="M9 6 V10" stroke="#7a5b00" stroke-width="1.6"/><circle cx="9" cy="12.5" r="1" fill="#7a5b00"/></svg>';
  return span;
}

function okIcon(): HTMLElement {
  const span = h("span", { class: "notice-icon", "aria-hidden": "true" });
  span.innerHTML =
    '<svg width="18" height="16" viewBox="0 0 18 16"><circle cx="9" cy="8" r="6.5" fill="none" stroke="#137a48" stroke-width="1.6"/><path d="M5.8 8.2 L8 10.3 L12.2 5.8" fill="none" stroke="#137a48" stroke-width="1.6"/></svg>';
  return span;
}

function gapWords(gap: number): string {
  if (!Number.isFinite(gap)) return "n/a";
  if (Math.abs(gap) < 0.0005) return "on the truth";
  return `${Math.abs(gap).toFixed(3)} ${gap < 0 ? "below" : "above"}`;
}

// Results screen ----------------------------------------------------------------

function runTag(v: EvalView): string {
  const p = v.params!;
  return `${fmtInt(p.nQueries)} queries · exploration ${p.exploration.toFixed(2)} · seed ${p.seed}`;
}

function renderResults(slide: boolean) {
  const v = state.sim;
  if (!v) return;
  const p = v.policies[state.policy]!;
  const truth = v.truth![state.policy].value;
  setText("results-title", `${p.name}: estimates against the truth`);
  setText("results-tag", runTag(v));

  const avail = ESTIMATORS.filter((e) => p.estimates[e].available && Number.isFinite(p.estimates[e].estimate));
  const gaps = avail.map((e) => ({ e, gap: p.estimates[e].estimate - truth }));
  gaps.sort((a, b) => Math.abs(a.gap) - Math.abs(b.gap));
  const covered = avail.filter((e) => p.estimates[e].ci_low <= truth && truth <= p.estimates[e].ci_high).length;
  const summary = $("results-summary");
  summary.replaceChildren(
    "The simulated online test gives this policy ",
    h("strong", {}, [`${fmt(truth)} clicks per query`]),
    `. Closest estimate: ${ESTIMATOR_NAMES[gaps[0].e]}, ${gapWords(gaps[0].gap)}. Farthest: ${ESTIMATOR_NAMES[gaps[gaps.length - 1].e]}, ${gapWords(gaps[gaps.length - 1].gap)}. ${covered} of ${avail.length} intervals contain the true value.`,
  );

  const spec: ForestSpec = {
    rows: estimatorRows(p, truth),
    truthLine: { value: truth, label: "True online value" },
    reference: { value: v.loggedMean, label: `Logged, current policy ${fmt(v.loggedMean)}` },
    axisLabel: "Expected clicks per query",
    title: `Estimates of ${p.name} against the true online value`,
    description: `True online value ${fmt(truth)}. ${ESTIMATORS.map((e) => `${ESTIMATOR_NAMES[e]} ${fmt(p.estimates[e].estimate)}`).join(", ")}.`,
  };
  if (slide) plots.results.update(spec);
  else {
    plots.results.render(spec, state.enter);
    state.enter = false;
  }
  estimateTable($<HTMLTableElement>("results-table"), p, truth);
}

$("download-results").addEventListener("click", () => {
  const v = state.sim;
  if (!v) return;
  const p = v.params!;
  const rows = [
    "policy_id,policy,estimator,estimate,ci_low,ci_high,effective_sample_size,true_value,gap_to_truth,clip,n_queries,exploration,seed",
  ];
  for (const id of POLICY_IDS) {
    const pv = v.policies[id]!;
    const t = v.truth![id].value;
    for (const est of ESTIMATORS) {
      const e = pv.estimates[est];
      rows.push(
        [id, pv.name, est, e.estimate, e.ci_low, e.ci_high, e.ess, t, e.estimate - t, v.clip ?? "none", p.nQueries, p.exploration, p.seed]
          .map(csvCell)
          .join(","),
      );
    }
  }
  download(`betterrankings-results-seed${p.seed}.csv`, rows.join("\n") + "\n");
});

$("download-log").addEventListener("click", async () => {
  const v = state.sim;
  if (!v) return;
  const res = await call({ type: "export-log" });
  if (res.type === "log-csv") download(`betterrankings-log-seed${v.params!.seed}.csv`, res.csv);
});

// Ranking screen -----------------------------------------------------------------

type Verdict = "worth" | "unclear" | "skip" | "na";

function verdict(e: Estimate, bar: number): Verdict {
  if (!e.available || !Number.isFinite(e.estimate)) return "na";
  if (e.ci_low > bar) return "worth";
  if (e.estimate > bar) return "unclear";
  return "skip";
}

const VERDICT_SHORT: Record<Verdict, string> = {
  worth: "✓ test it",
  unclear: "? unsure",
  skip: "✕ skip",
  na: "n/a",
};

const VERDICT_TEXT: Record<Verdict, string> = {
  worth: "✓ worth a live test",
  unclear: "? promising, unsure",
  skip: "✕ not worth it",
  na: "no estimate",
};

const segmented = $("rank-estimator");
for (const est of ESTIMATORS) {
  const input = h("input", { type: "radio", name: "rank-est", value: est });
  if (est === state.rankEst) (input as HTMLInputElement).checked = true;
  input.addEventListener("change", () => {
    state.rankEst = est;
    renderRanking(true);
  });
  segmented.append(h("label", {}, [input, h("span", {}, [ESTIMATOR_NAMES[est]])]));
}

function renderRanking(slide = false) {
  const v = state.sim;
  if (!v) return;
  const truth = v.truth!;
  const bar = v.loggedMean;
  const candidates = POLICY_IDS.filter((id) => id !== "current");
  const order = [...POLICY_IDS].sort((a, b) => truth[b].value - truth[a].value);
  const est = state.rankEst;

  const rows: ForestRow[] = order.map((id) => {
    const e = v.policies[id]!.estimates[est];
    const vd = id === "current" ? "na" : verdict(e, bar);
    return {
      id,
      label: v.policies[id]!.name,
      shortLabel: POLICY_BY_ID[id].short,
      value: e.available ? fmt(e.estimate) : "n/a",
      note: id === "current" ? "the current policy" : VERDICT_TEXT[vd],
      noteShort: id === "current" ? "current" : VERDICT_SHORT[vd],
      noteKind: vd === "worth" ? "good" : "plain",
      estimate: e.estimate,
      lo: e.ci_low,
      hi: e.ci_high,
      truth: truth[id].value,
      available: e.available && Number.isFinite(e.estimate),
      emphasis: vd === "worth",
    };
  });
  const hint = POLICY_IDS.map((id) => truth[id].value);
  const spec: ForestSpec = {
    rows,
    reference: { value: bar, label: `Bar to beat ${fmt(bar)}` },
    axisLabel: "Expected clicks per query",
    domainHint: hint,
    title: `Candidates ranked by true online value, with ${ESTIMATOR_NAMES[est]} estimates`,
    description: rows.map((r) => `${r.label}: true ${fmt(r.truth)}, estimate ${r.value}, ${r.note}`).join(". "),
  };
  if (slide) plots.ranking.update(spec);
  else plots.ranking.render(spec, false);

  const worth = candidates.filter((id) => verdict(v.policies[id]!.estimates[est], bar) === "worth");
  const trulyBetter = candidates.filter((id) => truth[id].value > truth.current.value);
  const names = (ids: string[]) => (ids.length ? ids.map((id) => POLICY_BY_ID[id as PolicyId].name).join(", ") : "none");
  const sameSet = worth.length === trulyBetter.length && worth.every((id) => trulyBetter.includes(id));
  $("ranking-summary").replaceChildren(
    `By ${ESTIMATOR_NAMES[est].toLowerCase()}, candidates whose whole interval clears the current policy's logged ${fmt(bar)} clicks per query: `,
    h("strong", {}, [names(worth)]),
    `. In the simulated online test, these truly beat the current policy: `,
    h("strong", {}, [names(trulyBetter)]),
    sameSet ? ". The two lists agree." : ". The lists differ, so this estimator would send the wrong set to live traffic.",
  );

  // Table
  const table = $<HTMLTableElement>("ranking-table");
  table.replaceChildren(
    h("thead", {}, [
      h("tr", {}, [
        h("th", { scope: "col" }, ["Policy"]),
        h("th", { scope: "col", class: "num" }, ["True value"]),
        h("th", { scope: "col", class: "num" }, ["Estimate"]),
        h("th", { scope: "col", class: "num" }, ["95% interval"]),
        h("th", { scope: "col" }, ["Verdict"]),
        h("th", { scope: "col" }, ["Truly beats current"]),
      ]),
    ]),
    h(
      "tbody",
      {},
      order.map((id) => {
        const e = v.policies[id]!.estimates[est];
        return h("tr", {}, [
          h("th", { scope: "row" }, [v.policies[id]!.name]),
          h("td", { class: "num" }, [fmt(truth[id].value)]),
          h("td", { class: "num" }, [fmt(e.estimate)]),
          h("td", { class: "num" }, [`${fmt(e.ci_low)} to ${fmt(e.ci_high)}`]),
          h("td", {}, [id === "current" ? "current policy" : VERDICT_TEXT[verdict(e, bar)]]),
          h("td", {}, [id === "current" ? "n/a" : truth[id].value > truth.current.value ? "yes" : "no"]),
        ]);
      }),
    ),
  );

  // Small multiples of rank agreement
  const grid = $("rank-grid");
  grid.replaceChildren();
  for (const k of ESTIMATORS) {
    const items = POLICY_IDS.map((id) => ({
      id,
      label: POLICY_BY_ID[id].short,
      truth: truth[id].value,
      estimate: v.policies[id]!.estimates[k].estimate,
    }));
    const bad = discordantPairs(
      items.map((i) => i.truth),
      items.map((i) => i.estimate),
    );
    const estOrder = [...items]
      .sort((a, b) => (Number.isFinite(b.estimate) ? b.estimate : -Infinity) - (Number.isFinite(a.estimate) ? a.estimate : -Infinity))
      .map((i) => i.label);
    const card = h("figure", { class: "rank-card" }, [
      h("h4", {}, [ESTIMATOR_NAMES[k]]),
      h("p", {}, [bad === 0 ? "Same order as the truth." : `${bad} of 15 pairs in the wrong order.`]),
      h("figcaption", { class: "sr-only" }, [`Estimated order: ${estOrder.join(", ")}.`]),
    ]);
    grid.append(card);
    card.append(rankChart(items, "Estimated order", Math.min(card.clientWidth, 420)));
  }
  observeWidth(grid, () => renderRanking());
}

// Diagnostics screen -------------------------------------------------------------

function renderDiagnostics() {
  const v = state.sim;
  if (!v) return;
  const p = v.policies[state.policy]!;
  renderDiagnosticsInto(v, p, {
    title: "diag-title",
    summary: "diag-summary",
    warnings: "diag-warnings",
    stats: "diag-stats",
    hist: "diag-hist",
    histTable: "diag-hist-table",
    zero: "diag-zero",
    positions: "diag-positions",
  });
}

interface DiagIds {
  title: string;
  summary: string;
  warnings: string;
  stats: string;
  hist: string;
  histTable: string;
  zero: string;
  positions: string;
}

function renderDiagnosticsInto(v: EvalView, p: PolicyView, ids: DiagIds) {
  const d = p.diagnostics;
  setText(ids.title, `Overlap diagnostics: ${p.name}`);
  setText(
    ids.summary,
    "Each logged query gets a weight: how likely this policy is to show the logged list, divided by how likely the current policy was. Weighting estimators are only as good as these weights.",
  );
  notices(
    $(ids.warnings),
    d,
    `Overlap looks healthy. The weights are worth about ${fmtInt(d.ess)} of ${fmtInt(d.n)} logged queries.`,
  );
  const stat = (label: string, value: string, note: string) =>
    h("div", {}, [h("dt", {}, [label]), h("dd", {}, [value, h("small", {}, [note])])]);
  const clipText = v.clip === null ? "no clip set" : `queries above clip ${v.clip}`;
  $(ids.stats).replaceChildren(
    stat("Effective sample size", fmtInt(d.ess), `of ${fmtInt(d.n)} queries (${fmtPct(d.ess_fraction, d.ess_fraction < 0.1 ? 1 : 0)})`),
    stat("Largest weight", fmtWeight(d.max_weight), `${fmtPct(d.max_weight_share, 1)} of all weight`),
    stat("Mean weight", Number.isFinite(d.mean_weight) ? d.mean_weight.toFixed(2) : "n/a", "close to 1 when the log covers this policy"),
    stat("Clipped", fmtInt(d.n_clipped), clipText),
  );

  const host = $(ids.hist);
  const draw = () =>
    host.replaceChildren(
      weightHistogram(p.histogram, v.clip, d.n_clipped, host.clientWidth, `Distribution of importance weights for ${p.name}`),
    );
  draw();
  observeWidth(host, draw);

  const hist = p.histogram;
  const pow = (l: number) => fmtWeight(Math.pow(10, l));
  $(ids.histTable).replaceChildren(
    h("thead", {}, [
      h("tr", {}, [h("th", { scope: "col" }, ["Weight range"]), h("th", { scope: "col", class: "num" }, ["Queries"])]),
    ]),
    h("tbody", {}, [
      h("tr", {}, [h("th", { scope: "row" }, ["0 (list never chosen by this policy)"]), h("td", { class: "num" }, [fmtInt(hist.zero)])]),
      ...hist.counts.map((c, i) =>
        h("tr", {}, [
          h("th", { scope: "row" }, [`${pow(hist.edges[i])} to ${pow(hist.edges[i + 1])}`]),
          h("td", { class: "num" }, [fmtInt(c)]),
        ]),
      ),
    ]),
  );
  setText(
    ids.zero,
    hist.zero
      ? `${fmtInt(hist.zero)} of ${fmtInt(hist.total)} logged queries have weight 0: this policy would never show those lists, so they add nothing to the weighting estimators. The chart shows the rest.`
      : "Every logged query has a positive weight: this policy could have shown each logged list.",
  );

  const maxRate = Math.max(...v.perPosition.map((r) => r.click_rate), 1e-9);
  $(ids.positions).replaceChildren(
    ...v.perPosition.map((r) => {
      const barEl = h("span", { class: "pos-bar" });
      barEl.style.width = `${(r.click_rate / maxRate) * 100}%`;
      return h("div", { class: "pos-row" }, [
        h("span", {}, [`Position ${r.position}`]),
        h("span", { class: "pos-track", "aria-hidden": "true" }, [barEl]),
        h("span", { class: "pos-value num" }, [fmtPct(r.click_rate, 1)]),
      ]);
    }),
  );
}

const widthObservers = new WeakMap<HTMLElement, { fn: () => void; w: number }>();
function observeWidth(host: HTMLElement, fn: () => void) {
  const existing = widthObservers.get(host);
  if (existing) {
    existing.fn = fn;
    existing.w = host.clientWidth;
    return;
  }
  const entry = { fn, w: host.clientWidth };
  widthObservers.set(host, entry);
  new ResizeObserver(() => {
    if (host.clientWidth !== entry.w && host.clientWidth > 0) {
      entry.w = host.clientWidth;
      entry.fn();
    }
  }).observe(host);
}

// Your own log -------------------------------------------------------------------

const fileInput = $<HTMLInputElement>("file");
const drop = $("drop");
const uploadTarget = $<HTMLSelectElement>("upload-target");
const uploadRun = $<HTMLButtonElement>("upload-run");
const uploadClip = $<HTMLInputElement>("upload-clip");

const UPLOAD_HINTS: Record<string, string> = {
  [TARGET_COLUMN]:
    "Uses your target_propensity column. The reward model and doubly robust estimators need the full target policy, so they are not available for it.",
};

function fillUploadTargets() {
  const keep = uploadTarget.value;
  uploadTarget.replaceChildren();
  if (state.upload.hasTarget)
    uploadTarget.append(h("option", { value: TARGET_COLUMN }, ["Your policy (target_propensity column)"]));
  for (const p of POLICIES) if (p.id !== "current") uploadTarget.append(h("option", { value: p.id }, [p.name]));
  if ([...uploadTarget.options].some((o) => o.value === keep)) uploadTarget.value = keep;
  updateUploadHint();
}

function updateUploadHint() {
  const id = uploadTarget.value;
  setText(
    "upload-target-hint",
    UPLOAD_HINTS[id] ??
      `${POLICY_BY_ID[id as PolicyId]?.description ?? ""} Built-in policies score items with the simulator's fixed weights.`,
  );
}

uploadTarget.addEventListener("change", updateUploadHint);
fillUploadTargets();

async function acceptFile(file: File | undefined) {
  if (!file) return;
  const msg = $("upload-message");
  msg.classList.remove("error");
  try {
    const text = await file.text();
    const header = text.slice(0, text.indexOf("\n") === -1 ? text.length : text.indexOf("\n"));
    state.upload.text = text;
    state.upload.fileName = file.name;
    state.upload.hasTarget = header.split(",").some((c) => c.trim().replace(/^"|"$/g, "") === TARGET_COLUMN);
    fillUploadTargets();
    if (state.upload.hasTarget) uploadTarget.value = TARGET_COLUMN;
    updateUploadHint();
    setText("file-status", `${file.name} · ${(file.size / 1024).toFixed(0)} KB, read in your browser`);
    setText("upload-message", "File ready. Pick a target policy and press Evaluate my log.");
    uploadRun.disabled = false;
  } catch {
    msg.classList.add("error");
    setText("upload-message", "That file could not be read. Check that it is a plain-text CSV and choose it again.");
  }
}

fileInput.addEventListener("change", () => void acceptFile(fileInput.files?.[0]));
for (const ev of ["dragenter", "dragover"])
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.add("is-over");
  });
for (const ev of ["dragleave", "drop"])
  drop.addEventListener(ev, (e) => {
    e.preventDefault();
    drop.classList.remove("is-over");
  });
drop.addEventListener("drop", (e) => void acceptFile((e as DragEvent).dataTransfer?.files?.[0]));

$("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!state.upload.text) return;
  const msg = $("upload-message");
  const bar = $<HTMLProgressElement>("upload-progress");
  msg.classList.remove("error");
  uploadRun.disabled = true;
  bar.hidden = false;
  bar.value = 0;
  setText("upload-message", "Reading your file…");
  try {
    const res = await call(
      {
        type: "upload",
        csv: state.upload.text,
        fileName: state.upload.fileName,
        target: uploadTarget.value as TargetId,
        clip: CLIP_LEVELS[state.upload.clipIdx],
        nBoot: DEFAULTS.nBoot,
      },
      (stage, f) => {
        bar.value = f;
        setText("upload-message", `${stage}…`);
      },
    );
    if (res.type !== "result") throw new Error("Unexpected reply from the worker.");
    state.upload.view = res.view;
    state.upload.enter = true;
    setText("upload-message", `Done. Evaluated ${fmtInt(res.view.n)} queries from ${state.upload.fileName}.`);
    renderUpload(false);
  } catch (err) {
    msg.classList.add("error");
    setText("upload-message", `${(err as Error).message} Fix the file and choose it again, or start from the sample CSV.`);
  } finally {
    uploadRun.disabled = false;
    bar.hidden = true;
  }
});

uploadClip.value = String(state.upload.clipIdx);
syncFill(uploadClip);
clipControl(
  uploadClip,
  "upload-clip-out",
  "upload",
  () => state.upload.view,
  (policies, c) => {
    const v = state.upload.view;
    if (!v) return;
    v.policies = policies;
    v.clip = c;
    state.upload.clipIdx = Number(uploadClip.value);
    renderUpload(true);
  },
);

function renderUpload(slide: boolean) {
  const v = state.upload.view;
  if (!v) return;
  $("upload-results").hidden = false;
  const [id] = Object.keys(v.policies) as TargetId[];
  const p = v.policies[id]!;
  setText("upload-results-title", `Estimates for ${p.name}`);
  $("upload-summary").replaceChildren(
    `${v.fileName}: ${fmtInt(v.n)} queries with lists of ${v.nSlots}, `,
    h("strong", {}, [`${fmt(v.loggedMean)} clicks per query`]),
    " under the logging policy. Your own log has no known true value, so there is no truth line: compare the estimates with each other and with the logged rate.",
  );
  notices($("upload-warnings"), p.diagnostics);
  const spec: ForestSpec = {
    rows: estimatorRows(p, null),
    reference: { value: v.loggedMean, label: `Logged ${fmt(v.loggedMean)}` },
    axisLabel: "Expected clicks per query",
    title: `Estimates of ${p.name} from your log`,
    description: ESTIMATORS.map((e) => `${ESTIMATOR_NAMES[e]} ${fmt(p.estimates[e].estimate)}`).join(", "),
  };
  if (slide) plots.upload.update(spec);
  else {
    plots.upload.render(spec, state.upload.enter);
    state.upload.enter = false;
  }
  estimateTable($<HTMLTableElement>("upload-table"), p, null);
}

$("upload-download").addEventListener("click", () => {
  const v = state.upload.view;
  if (!v) return;
  const rows = ["file,policy_id,policy,estimator,estimate,ci_low,ci_high,effective_sample_size,clip,n_queries,logged_clicks_per_query"];
  for (const [id, p] of Object.entries(v.policies)) {
    for (const est of ESTIMATORS) {
      const e = p!.estimates[est];
      rows.push(
        [v.fileName, id, p!.name, est, e.available ? e.estimate : null, e.ci_low, e.ci_high, e.ess, v.clip ?? "none", v.n, v.loggedMean]
          .map(csvCell)
          .join(","),
      );
    }
  }
  const base = (v.fileName ?? "log").replace(/\.csv$/i, "");
  download(`${base}-betterrankings-results.csv`, rows.join("\n") + "\n");
});

// Start ----------------------------------------------------------------------------

updatePolicyHint();
setText("clip-out", fmtClip(CLIP_LEVELS[state.clipIdx]));
setText("upload-clip-out", fmtClip(CLIP_LEVELS[state.upload.clipIdx]));
void runEvaluation();
