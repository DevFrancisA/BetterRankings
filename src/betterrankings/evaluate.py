"""Putting the pieces together: evaluate policies on a log and check against the truth."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import sqlstore
from .estimators import (
    ESTIMATORS,
    Estimate,
    all_estimates,
    bootstrap_indices,
    overlap_diagnostics,
)
from .logs import LogData, LogFormatError, simulate_log
from .policies import POLICY_IDS, POLICY_NAMES, expected_clicks, policy_probs
from .reward_model import CrossFit, cross_fit
from .world import WorldConfig, make_catalog, make_params, sample_queries, true_click_matrix

TARGET_COLUMN = "target_propensity"
DEFAULTS = {
    "n_queries": 5000,
    "exploration": 0.2,
    "seed": 7,
    "clip": 20.0,
    "n_boot": 200,
    "n_folds": 2,
    "n_online": 20000,
}


@dataclass
class PolicyResult:
    policy_id: str
    name: str
    estimates: dict[str, Estimate]
    diagnostics: dict
    weights: np.ndarray = field(repr=False)
    dm: np.ndarray | None = field(repr=False)


@dataclass
class Evaluation:
    n: int
    logged_mean: float
    clip: float | None
    policies: dict[str, PolicyResult]
    per_position: pd.DataFrame

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "logged_mean": self.logged_mean,
            "clip": self.clip,
            "per_position": self.per_position.to_dict(orient="records"),
            "policies": {
                pid: {
                    "name": r.name,
                    "estimates": {k: e.to_dict() for k, e in r.estimates.items()},
                    "diagnostics": r.diagnostics,
                }
                for pid, r in self.policies.items()
            },
        }


def evaluate_log(
    log: LogData,
    policy_ids: list[str] | tuple[str, ...] = POLICY_IDS,
    *,
    cfg: WorldConfig = WorldConfig(),
    exploration: float = DEFAULTS["exploration"],
    clip: float | None = DEFAULTS["clip"],
    n_boot: int = DEFAULTS["n_boot"],
    seed: int = 0,
    n_folds: int = DEFAULTS["n_folds"],
    per_position: pd.DataFrame | None = None,
) -> Evaluation:
    """Estimate each policy's expected clicks per query from a log."""
    f = log.features
    if f.ctx.shape[1] != cfg.d_ctx or f.feats.shape[2] != cfg.d_item:
        raise LogFormatError(
            f"built-in policies need {cfg.d_ctx} context and {cfg.d_item} item features, "
            f"the log has {f.ctx.shape[1]} and {f.feats.shape[2]}"
        )
    params = make_params(cfg)
    idx = bootstrap_indices(len(log), n_boot, seed)
    fit: CrossFit | None = None
    results: dict[str, PolicyResult] = {}
    for pid in policy_ids:
        if pid == TARGET_COLUMN:
            if log.target_propensity is None:
                raise LogFormatError("the log has no target_propensity column")
            pi_logged, dm, q_logged = log.target_propensity, None, None
            name = "Your policy (target_propensity column)"
        else:
            if fit is None:
                fit = cross_fit(f, log.clicks, n_folds, seed)
            probs = policy_probs(pid, f, cfg, params, exploration)
            pi_logged = probs[:, 0]  # columns follow the shown order, so ranking 0 is the logged one
            dm = expected_clicks(probs, fit.q)
            q_logged = fit.logged()
            name = POLICY_NAMES[pid]
        w = pi_logged / log.propensity
        results[pid] = PolicyResult(
            policy_id=pid,
            name=name,
            estimates=all_estimates(log.rewards, log.propensity, pi_logged, dm, q_logged, clip, idx),
            diagnostics=overlap_diagnostics(w, clip),
            weights=w,
            dm=dm,
        )
    return Evaluation(
        n=len(log),
        logged_mean=float(log.rewards.mean()),
        clip=clip,
        policies=results,
        per_position=per_position if per_position is not None else pd.DataFrame(),
    )


def evaluate_frame(df: pd.DataFrame, policy_ids=POLICY_IDS, *, cfg: WorldConfig = WorldConfig(), **kw) -> Evaluation:
    """Load a wide log table through SQLite, then evaluate."""
    conn = sqlstore.load_log(df, expect_ctx=cfg.d_ctx, expect_item=cfg.d_item)
    try:
        log = sqlstore.log_data(conn)
        table = sqlstore.per_position_clicks(conn)
    finally:
        conn.close()
    return evaluate_log(log, policy_ids, cfg=cfg, per_position=table, **kw)


def ground_truth(
    cfg: WorldConfig,
    seed: int,
    exploration: float,
    policy_ids=POLICY_IDS,
    n_online: int = DEFAULTS["n_online"],
) -> dict[str, dict]:
    """A simulated online test: exact expected clicks on fresh traffic for each policy."""
    params = make_params(cfg)
    catalog = make_catalog(cfg, seed)
    f, _ = sample_queries(cfg, catalog, n_online, seed, "online")
    g = true_click_matrix(cfg, params, f)
    out = {}
    for pid in policy_ids:
        v = expected_clicks(policy_probs(pid, f, cfg, params, exploration), g)
        out[pid] = {"value": float(v.mean()), "se": float(v.std(ddof=1) / np.sqrt(len(v)))}
    return out


@dataclass
class Comparison:
    log: pd.DataFrame
    evaluation: Evaluation
    truth: dict[str, dict]

    def table(self) -> pd.DataFrame:
        rows = []
        for pid, r in self.evaluation.policies.items():
            row = {"policy": pid, "truth": self.truth[pid]["value"]}
            for est in ESTIMATORS:
                row[est] = r.estimates[est].estimate
            rows.append(row)
        return pd.DataFrame(rows).set_index("policy")

    def rank_agreement(self) -> dict[str, int]:
        """Pairs of policies each estimator puts in the wrong order (0 is perfect)."""
        t = self.table()
        out = {}
        for est in ESTIMATORS:
            out[est] = discordant_pairs(t["truth"].to_numpy(), t[est].to_numpy())
        return out


def discordant_pairs(truth: np.ndarray, est: np.ndarray) -> int:
    bad = 0
    for i in range(len(truth)):
        for j in range(i + 1, len(truth)):
            if not np.isfinite(est[i]) or not np.isfinite(est[j]):
                bad += 1
            elif (truth[i] - truth[j]) * (est[i] - est[j]) < 0:
                bad += 1
    return bad


def run_comparison(
    cfg: WorldConfig = WorldConfig(),
    *,
    n_queries: int = DEFAULTS["n_queries"],
    exploration: float = DEFAULTS["exploration"],
    seed: int = DEFAULTS["seed"],
    clip: float | None = DEFAULTS["clip"],
    n_boot: int = DEFAULTS["n_boot"],
    n_folds: int = DEFAULTS["n_folds"],
    n_online: int = DEFAULTS["n_online"],
    policy_ids=POLICY_IDS,
) -> Comparison:
    """Simulate a log, estimate every policy from it, and run the online test."""
    df = simulate_log(cfg, n_queries, exploration, seed)
    ev = evaluate_frame(
        df, policy_ids, cfg=cfg, exploration=exploration, clip=clip, n_boot=n_boot, seed=seed, n_folds=n_folds
    )
    truth = ground_truth(cfg, seed, exploration, policy_ids, n_online)
    return Comparison(df, ev, truth)

