"""SQLite log layer.

A log table is loaded into an in-memory SQLite database and every aggregation
the estimators need comes from the queries in ``sql/``.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from importlib import resources

import numpy as np
import pandas as pd

from .logs import LogData, LogFormatError, detect_shape, validate_log
from .world import Features


@lru_cache(maxsize=None)
def sql_text(name: str) -> str:
    return resources.files("betterrankings").joinpath("sql", f"{name}.sql").read_text(encoding="utf-8")


def run_query(conn: sqlite3.Connection, name: str, **params: object) -> pd.DataFrame:
    return pd.read_sql_query(sql_text(name), conn, params=params or None)


def load_log(df: pd.DataFrame, expect_ctx: int | None = None, expect_item: int | None = None) -> sqlite3.Connection:
    """Validate a wide log table and load it into a fresh in-memory database."""
    validate_log(df, expect_ctx, expect_item)
    n_slots, d_ctx, d_item = detect_shape(list(df.columns))
    n = len(df)
    rows = np.arange(n)
    conn = sqlite3.connect(":memory:")
    conn.executescript(sql_text("schema"))

    target = df["target_propensity"].astype(float).tolist() if "target_propensity" in df.columns else [None] * n
    conn.executemany(
        "INSERT INTO queries VALUES (?, ?, ?, ?)",
        zip(rows.tolist(), df["query_id"].astype(str).tolist(), df["propensity"].astype(float).tolist(), target),
    )
    ctx = df[[f"ctx_{a}" for a in range(1, d_ctx + 1)]].to_numpy(dtype=float)
    conn.executemany(
        "INSERT INTO contexts VALUES (?, ?, ?)",
        ((int(i), a + 1, float(ctx[i, a])) for i in range(n) for a in range(d_ctx)),
    )
    imp, feat = [], []
    for k in range(1, n_slots + 1):
        ids = df[f"item_{k}"].astype(str).tolist()
        pop = df[f"item_{k}_pop"].astype(float).tolist()
        clk = df[f"click_{k}"].astype(int).tolist()
        imp.extend(zip(rows.tolist(), [k] * n, ids, pop, clk))
        for b in range(1, d_item + 1):
            feat.extend(zip(rows.tolist(), [k] * n, [b] * n, df[f"item_{k}_f{b}"].astype(float).tolist()))
    conn.executemany("INSERT INTO impressions VALUES (?, ?, ?, ?, ?)", imp)
    conn.executemany("INSERT INTO item_features VALUES (?, ?, ?, ?)", feat)
    conn.commit()
    return conn


def log_data(conn: sqlite3.Connection) -> LogData:
    """Arrays for the estimators, built from the SQL aggregations."""
    per_query = run_query(conn, "per_query_rewards")
    n = len(per_query)
    imp = run_query(conn, "impression_matrix")
    if len(imp) % n:
        raise LogFormatError("every query must show the same number of items")
    n_slots = len(imp) // n
    ctx_long = run_query(conn, "context_matrix")
    feat_long = run_query(conn, "item_feature_matrix")
    d_ctx = len(ctx_long) // n
    d_item = len(feat_long) // (n * n_slots)

    features = Features(
        ctx=ctx_long["value"].to_numpy(dtype=float).reshape(n, d_ctx),
        pop=imp["pop"].to_numpy(dtype=float).reshape(n, n_slots),
        feats=feat_long["value"].to_numpy(dtype=float).reshape(n, n_slots, d_item),
    )
    target = per_query["target_propensity"]
    return LogData(
        query_id=per_query["query_id"].to_numpy(dtype=str),
        features=features,
        item_ids=imp["item_id"].to_numpy(dtype=str).reshape(n, n_slots),
        clicks=imp["clicked"].to_numpy(dtype=np.int64).reshape(n, n_slots),
        rewards=per_query["clicks"].to_numpy(dtype=float),
        propensity=per_query["propensity"].to_numpy(dtype=float),
        target_propensity=None if target.isna().all() else target.to_numpy(dtype=float),
    )


def per_position_clicks(conn: sqlite3.Connection) -> pd.DataFrame:
    return run_query(conn, "per_position_clicks")


def item_click_rates(conn: sqlite3.Connection, min_impressions: int = 20, limit: int = 10) -> pd.DataFrame:
    return run_query(conn, "item_click_rates", min_impressions=min_impressions, limit=limit)


def log_summary(conn: sqlite3.Connection) -> dict:
    return run_query(conn, "log_summary").iloc[0].to_dict()
