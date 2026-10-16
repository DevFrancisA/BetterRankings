// Messages between the page and the computation worker.

import type { Diagnostics, Estimate, EstimatorId } from "./core/estimators";
import type { TargetId, Truth } from "./core/evaluate";
import type { PositionRow } from "./core/logTable";

export type Session = "sim" | "upload";

export interface RunParams {
  nQueries: number;
  exploration: number;
  seed: number;
  clip: number | null;
  nBoot: number;
}

export interface Histogram {
  /** log10 bin edges for positive weights. */
  edges: number[];
  counts: number[];
  zero: number;
  total: number;
}

export interface PolicyView {
  id: TargetId;
  name: string;
  estimates: Record<EstimatorId, Estimate>;
  diagnostics: Diagnostics;
  histogram: Histogram;
}

export interface EvalView {
  session: Session;
  n: number;
  nSlots: number;
  loggedMean: number;
  clip: number | null;
  perPosition: PositionRow[];
  policies: Partial<Record<TargetId, PolicyView>>;
  truth: Record<string, Truth> | null;
  params: RunParams | null;
  fileName?: string;
  hasTargetColumn?: boolean;
}

export type Request =
  | { type: "run"; id: number; params: RunParams }
  | { type: "reclip"; id: number; session: Session; clip: number | null }
  | { type: "upload"; id: number; csv: string; fileName: string; target: TargetId; clip: number | null; nBoot: number }
  | { type: "export-log"; id: number };

export type Response =
  | { type: "progress"; id: number; stage: string; fraction: number }
  | { type: "result"; id: number; view: EvalView }
  | { type: "reclipped"; id: number; session: Session; clip: number | null; policies: EvalView["policies"] }
  | { type: "log-csv"; id: number; csv: string }
  | { type: "error"; id: number; message: string };

/** Clip levels on the slider; null means no clipping. */
export const CLIP_LEVELS: (number | null)[] = [1, 1.5, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 200, 500, null];
