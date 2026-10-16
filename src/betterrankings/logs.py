"""Log tables: simulating them and reading them from CSV.

One row per query, positions numbered from 1 (the top of the list)::

    query_id                      any identifier
    ctx_1 .. ctx_D                query context features
    item_K                        id of the item shown at position K
    click_K                       1 if that item was clicked, else 0
    item_K_pop                    popularity feature of that item
    item_K_f1 .. item_K_fF        other item features
    propensity                    probability the logging policy gave the
                                  whole ranking that was shown
    target_propensity (optional)  probability your own target policy gives
                                  the ranking that was shown

The set of shown items is treated as the candidate set, so a policy re-ranks
exactly those items. Rows are kept in file order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .perms import MAX_SLOTS, permutations
from .policies import policy_probs
from .rng import Rng
from .world import (
    Features,
    WorldConfig,
    make_catalog,
    make_params,
    sample_queries,
    true_relevance,
)


@dataclass
class LogData:
    query_id: np.ndarray  # (n,) strings
    features: Features  # candidates in shown order
    item_ids: np.ndarray  # (n, m) strings
    clicks: np.ndarray  # (n, m) 0/1
    rewards: np.ndarray  # (n,) total clicks per query
    propensity: np.ndarray  # (n,)
    target_propensity: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.propensity)

    @property
    def n_slots(self) -> int:
        return self.clicks.shape[1]


def log_columns(n_slots: int, d_ctx: int, d_item: int, target: bool = False) -> list[str]:
    cols = ["query_id"] + [f"ctx_{a}" for a in range(1, d_ctx + 1)]
    cols += [f"item_{k}" for k in range(1, n_slots + 1)]
    cols += [f"click_{k}" for k in range(1, n_slots + 1)]
    for k in range(1, n_slots + 1):
        cols += [f"item_{k}_pop"] + [f"item_{k}_f{b}" for b in range(1, d_item + 1)]
    cols.append("propensity")
    if target:
        cols.append("target_propensity")
    return cols


def simulate_log(cfg: WorldConfig, n_queries: int, exploration: float, seed: int) -> pd.DataFrame:
    """Run the logging policy on simulated traffic and record what happened."""
    if not 0.0 <= exploration <= 1.0:
        raise ValueError("exploration must be between 0 and 1")
    params = make_params(cfg)
    catalog = make_catalog(cfg, seed)
    f, ids = sample_queries(cfg, catalog, n_queries, seed, "log")
    probs = policy_probs("current", f, cfg, params, exploration)

    u = Rng(seed, "log-rankings").uniforms(n_queries)
    cdf = np.cumsum(probs, axis=1)
    chosen = np.minimum((cdf <= (u * cdf[:, -1])[:, None]).sum(axis=1), probs.shape[1] - 1)
    propensity = probs[np.arange(n_queries), chosen]
    order = permutations(cfg.n_slots)[chosen]  # slot shown at each position

    rows = np.arange(n_queries)[:, None]
    shown = Features(f.ctx, f.pop[rows, order], f.feats[rows, order])
    shown_ids = ids[rows, order]
    p_click = true_relevance(cfg, params, shown) * cfg.position_weights()[None, :]
    u_click = Rng(seed, "log-clicks").uniforms(n_queries * cfg.n_slots).reshape(p_click.shape)
    clicks = (u_click < p_click).astype(np.int64)

    data: dict[str, np.ndarray] = {"query_id": np.arange(1, n_queries + 1)}
    for a in range(cfg.d_ctx):
        data[f"ctx_{a + 1}"] = shown.ctx[:, a]
    for k in range(cfg.n_slots):
        data[f"item_{k + 1}"] = shown_ids[:, k]
    for k in range(cfg.n_slots):
        data[f"click_{k + 1}"] = clicks[:, k]
    for k in range(cfg.n_slots):
        data[f"item_{k + 1}_pop"] = shown.pop[:, k]
        for b in range(cfg.d_item):
            data[f"item_{k + 1}_f{b + 1}"] = shown.feats[:, k, b]
    data["propensity"] = propensity
    return pd.DataFrame(data, columns=log_columns(cfg.n_slots, cfg.d_ctx, cfg.d_item))


def write_log_csv(df: pd.DataFrame, path: str | Path) -> None:
    # 17 significant digits round-trip every float exactly, so the browser
    # reads back the same numbers Python wrote.
    df.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def read_log_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"query_id": str}, float_precision="round_trip")


class LogFormatError(ValueError):
    pass


def detect_shape(columns: list[str]) -> tuple[int, int, int]:
    """Number of list slots, context features and item features in a log table."""
    cols = set(columns)
    n_slots = 0
    while f"item_{n_slots + 1}" in cols:
        n_slots += 1
    d_ctx = 0
    while f"ctx_{d_ctx + 1}" in cols:
        d_ctx += 1
    d_item = 0
    while f"item_1_f{d_item + 1}" in cols:
        d_item += 1
    return n_slots, d_ctx, d_item


def validate_log(df: pd.DataFrame, expect_ctx: int | None = None, expect_item: int | None = None) -> None:
    """Raise LogFormatError with a plain explanation when a log table is unusable."""
    n_slots, d_ctx, d_item = detect_shape(list(df.columns))
    if len(df) == 0:
        raise LogFormatError("the log has no rows")
    if not 2 <= n_slots <= MAX_SLOTS:
        raise LogFormatError(f"expected item_1 .. item_K columns with K between 2 and {MAX_SLOTS}")
    if expect_ctx is not None and d_ctx != expect_ctx:
        raise LogFormatError(f"expected {expect_ctx} context columns ctx_1 .. ctx_{expect_ctx}, found {d_ctx}")
    if expect_item is not None and d_item != expect_item:
        raise LogFormatError(f"expected {expect_item} item features item_K_f1 .. item_K_f{expect_item}, found {d_item}")
    missing = [c for c in log_columns(n_slots, d_ctx, d_item) if c not in df.columns]
    if missing:
        raise LogFormatError("missing columns: " + ", ".join(missing[:8]))
    numeric = [c for c in log_columns(n_slots, d_ctx, d_item) if re.match(r"^(ctx_|click_|propensity|item_\d+_)", c)]
    bad = [c for c in numeric if not pd.api.types.is_numeric_dtype(df[c]) or df[c].isna().any()]
    if bad:
        raise LogFormatError("these columns must be numbers with no blanks: " + ", ".join(bad[:8]))
    clicks = df[[f"click_{k}" for k in range(1, n_slots + 1)]].to_numpy()
    if not np.isin(clicks, (0, 1)).all():
        raise LogFormatError("click columns must contain only 0 or 1")
    p = df["propensity"].to_numpy(dtype=float)
    if (p <= 0).any() or (p > 1).any():
        raise LogFormatError("propensity must be greater than 0 and at most 1")
    if "target_propensity" in df.columns:
        t = df["target_propensity"].to_numpy(dtype=float)
        if np.isnan(t).any() or (t < 0).any() or (t > 1).any():
            raise LogFormatError("target_propensity must be between 0 and 1 with no blanks")
