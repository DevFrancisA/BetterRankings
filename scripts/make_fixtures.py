"""Write the shared fixtures the TypeScript port is checked against.

    python scripts/make_fixtures.py

Writes fixtures/*.csv logs, fixtures/expected.json with the Python results,
and web/public/sample-log.csv (the sample offered on the site).
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from betterrankings.estimators import ESTIMATORS
from betterrankings.evaluate import evaluate_frame, ground_truth
from betterrankings.logs import read_log_csv, simulate_log, write_log_csv
from betterrankings.policies import POLICY_IDS, policy_probs
from betterrankings.rng import Rng, derive_seed
from betterrankings.world import WorldConfig, make_params

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures"
SAMPLE = ROOT / "web" / "public" / "sample-log.csv"

SIM_CASES = [
    {"name": "four_slots", "n_queries": 1200, "exploration": 0.2, "seed": 11, "n_slots": 4,
     "n_boot": 100, "n_folds": 2, "n_online": 3000, "clips": [20.0, 5.0, None]},
    {"name": "five_slots_low_exploration", "n_queries": 600, "exploration": 0.05, "seed": 5, "n_slots": 5,
     "n_boot": 60, "n_folds": 2, "n_online": 1500, "clips": [10.0]},
]
SAMPLE_CASE = {"name": "sample_upload", "n_queries": 400, "exploration": 0.3, "seed": 1, "n_slots": 4,
               "n_boot": 80, "n_folds": 2, "eval_seed": 0, "clips": [20.0]}


def _clean(x):
    """JSON-safe: NaN and infinity become null."""
    if isinstance(x, dict):
        return {k: _clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_clean(v) for v in x]
    if isinstance(x, (np.floating, float)):
        return None if not math.isfinite(float(x)) else float(x)
    if isinstance(x, np.integer):
        return int(x)
    return x


def _evaluation(df, cfg, case, clip, seed, policy_ids):
    ev = evaluate_frame(df, policy_ids, cfg=cfg, exploration=case["exploration"], clip=clip,
                        n_boot=case["n_boot"], seed=seed, n_folds=case["n_folds"])
    return {
        "clip": clip,
        "logged_mean": ev.logged_mean,
        "per_position": ev.per_position.to_dict(orient="records"),
        "policies": {
            pid: {
                "estimates": {k: r.estimates[k].to_dict() for k in ESTIMATORS},
                "diagnostics": {k: v for k, v in r.diagnostics.items() if k != "warnings"}
                | {"warning_codes": [w["code"] for w in r.diagnostics["warnings"]]},
            }
            for pid, r in ev.policies.items()
        },
    }


def sample_frame():
    """The downloadable sample: a simulated log plus an example target_propensity column."""
    case = SAMPLE_CASE
    cfg = replace(WorldConfig(), n_slots=case["n_slots"])
    df = simulate_log(cfg, case["n_queries"], case["exploration"], case["seed"])
    conn_free = df.copy()
    from betterrankings import sqlstore

    log = sqlstore.log_data(sqlstore.load_log(conn_free))
    probs = policy_probs("relevance_soft", log.features, cfg, make_params(cfg), case["exploration"])
    df["target_propensity"] = probs[:, 0]
    return df


def build() -> dict:
    FIXTURES.mkdir(exist_ok=True)
    rng = Rng(12345)
    out = {
        "rng": {
            "seed": 12345,
            "uniforms": rng.uniforms(8).tolist(),
            "derived": [{"seed": s, "name": n, "value": derive_seed(s, n)}
                        for s, n in [(7, "log-clicks"), (0, "bootstrap"), (4294967295, "catalog")]],
            "normals": Rng(3, "normals").normals(6).tolist(),
        },
        "cases": [],
    }
    for case in SIM_CASES:
        cfg = replace(WorldConfig(), n_slots=case["n_slots"])
        df = simulate_log(cfg, case["n_queries"], case["exploration"], case["seed"])
        log_file = f"{case['name']}.csv"
        write_log_csv(df, FIXTURES / log_file)
        df = read_log_csv(FIXTURES / log_file)
        out["cases"].append({
            "name": case["name"],
            "log": log_file,
            "simulated": True,
            "config": {k: v for k, v in case.items() if k not in ("name", "clips")},
            "truth": ground_truth(cfg, case["seed"], case["exploration"], POLICY_IDS, case["n_online"]),
            "evaluations": [_evaluation(df, cfg, case, c, case["seed"], POLICY_IDS) for c in case["clips"]],
        })

    case = SAMPLE_CASE
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    write_log_csv(sample_frame(), SAMPLE)
    df = read_log_csv(SAMPLE)
    cfg = replace(WorldConfig(), n_slots=case["n_slots"])
    ids = ["target_propensity", *[p for p in POLICY_IDS if p != "current"]]
    out["cases"].append({
        "name": case["name"],
        "log": "../web/public/sample-log.csv",
        "simulated": False,
        "config": {k: v for k, v in case.items() if k not in ("name", "clips")},
        "evaluations": [_evaluation(df, cfg, case, c, case["eval_seed"], ids) for c in case["clips"]],
    })
    return _clean(out)


def main() -> int:
    data = build()
    (FIXTURES / "expected.json").write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote {len(data['cases'])} fixture cases to {FIXTURES}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
