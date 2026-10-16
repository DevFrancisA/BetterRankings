// All heavy computation runs here so the page stays responsive.

import { DEFAULTS, evaluateLog, groundTruth, reclip, TARGET_COLUMN, type Evaluation } from "./core/evaluate";
import { LogFormatError, logToCsv, parseLogCsv, simulateLog, type LogTable } from "./core/logTable";
import { POLICY_IDS } from "./core/policies";
import { DEFAULT_WORLD } from "./core/world";
import { type EvalView, type Histogram, type Request, type Response, type Session } from "./protocol";

const scope = self as unknown as {
  postMessage(msg: Response): void;
  onmessage: ((e: MessageEvent<Request>) => void) | null;
};

const sessions: Partial<Record<Session, Evaluation>> = {};
let simLog: LogTable | null = null;

function histogram(w: Float64Array): Histogram {
  const BIN = 0.2;
  const FLOOR = -3;
  let lo = Infinity;
  let hi = -Infinity;
  let zero = 0;
  for (const x of w) {
    if (x <= 0) {
      zero++;
      continue;
    }
    const l = Math.max(FLOOR, Math.log10(x));
    if (l < lo) lo = l;
    if (l > hi) hi = l;
  }
  if (!Number.isFinite(lo)) return { edges: [], counts: [], zero, total: w.length };
  const start = Math.floor(lo / BIN - 1e-9) * BIN;
  const nb = Math.max(1, Math.ceil((hi - start) / BIN + 1e-9));
  const counts = new Array(nb).fill(0);
  for (const x of w) {
    if (x <= 0) continue;
    const l = Math.max(FLOOR, Math.log10(x));
    counts[Math.min(nb - 1, Math.floor((l - start) / BIN))]++;
  }
  const edges = Array.from({ length: nb + 1 }, (_, i) => +(start + i * BIN).toFixed(6));
  return { edges, counts, zero, total: w.length };
}

function policiesView(ev: Evaluation): EvalView["policies"] {
  const out: EvalView["policies"] = {};
  for (const p of Object.values(ev.policies)) {
    if (!p) continue;
    out[p.id] = {
      id: p.id,
      name: p.name,
      estimates: p.estimates,
      diagnostics: p.diagnostics,
      histogram: histogram(p.weights),
    };
  }
  return out;
}

function post(msg: Response) {
  scope.postMessage(msg);
}

scope.onmessage = (e) => {
  const req = e.data;
  try {
    if (req.type === "run") {
      const p = req.params;
      const progress = (stage: string, fraction: number) => post({ type: "progress", id: req.id, stage, fraction });
      progress("Simulating logged traffic", 0.02);
      const cfg = DEFAULT_WORLD;
      simLog = simulateLog(cfg, p.nQueries, p.exploration, p.seed);
      const ev = evaluateLog(simLog, POLICY_IDS, {
        cfg,
        exploration: p.exploration,
        clip: p.clip,
        nBoot: p.nBoot,
        seed: p.seed,
        nFolds: DEFAULTS.nFolds,
        onProgress: (s, f) => progress(s, 0.05 + f * 0.75),
      });
      progress("Running the simulated online test", 0.85);
      const truth = groundTruth(cfg, p.seed, p.exploration, POLICY_IDS, DEFAULTS.nOnline);
      sessions.sim = ev;
      post({
        type: "result",
        id: req.id,
        view: {
          session: "sim",
          n: ev.n,
          nSlots: cfg.nSlots,
          loggedMean: ev.loggedMean,
          clip: ev.clip,
          perPosition: ev.perPosition,
          policies: policiesView(ev),
          truth,
          params: p,
        },
      });
    } else if (req.type === "reclip") {
      const ev = sessions[req.session];
      if (!ev) throw new Error("Run an evaluation first.");
      reclip(ev, req.clip);
      post({ type: "reclipped", id: req.id, session: req.session, clip: req.clip, policies: policiesView(ev) });
    } else if (req.type === "upload") {
      post({ type: "progress", id: req.id, stage: "Reading your file", fraction: 0.03 });
      const log = parseLogCsv(req.csv, { expectCtx: DEFAULT_WORLD.dCtx, expectItem: DEFAULT_WORLD.dItem });
      if (req.target === TARGET_COLUMN && !log.targetPropensity) {
        throw new LogFormatError(
          "Your file has no target_propensity column. Add one, or pick a built-in policy to evaluate.",
        );
      }
      const ev = evaluateLog(log, [req.target], {
        cfg: { ...DEFAULT_WORLD, nSlots: log.features.m },
        exploration: DEFAULTS.exploration,
        clip: req.clip,
        nBoot: req.nBoot,
        seed: 0,
        nFolds: DEFAULTS.nFolds,
        onProgress: (stage, fraction) => post({ type: "progress", id: req.id, stage, fraction: 0.05 + fraction * 0.9 }),
      });
      sessions.upload = ev;
      post({
        type: "result",
        id: req.id,
        view: {
          session: "upload",
          n: ev.n,
          nSlots: log.features.m,
          loggedMean: ev.loggedMean,
          clip: ev.clip,
          perPosition: ev.perPosition,
          policies: policiesView(ev),
          truth: null,
          params: null,
          fileName: req.fileName,
          hasTargetColumn: log.targetPropensity !== null,
        },
      });
    } else if (req.type === "export-log") {
      if (!simLog) throw new Error("Run an evaluation first.");
      post({ type: "log-csv", id: req.id, csv: logToCsv(simLog) });
    }
  } catch (err) {
    const message =
      err instanceof LogFormatError
        ? err.message
        : `Something went wrong while computing: ${err instanceof Error ? err.message : String(err)}`;
    post({ type: "error", id: req.id, message });
  }
};
