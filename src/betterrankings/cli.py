"""Command line interface: simulate, evaluate, compare."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace

import pandas as pd

from . import sqlstore
from .estimators import ESTIMATOR_NAMES, ESTIMATORS
from .evaluate import DEFAULTS, TARGET_COLUMN, evaluate_log, run_comparison
from .logs import LogFormatError, read_log_csv, simulate_log, write_log_csv
from .policies import POLICIES, POLICY_IDS
from .world import WorldConfig


def _clip(text: str) -> float | None:
    if text.lower() in ("none", "off", "inf"):
        return None
    value = float(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("clip must be positive, or 'none'")
    return value


def _fmt(x: float) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.4f}"


def _estimate_rows(ev, policy_id: str, truth: float | None = None) -> pd.DataFrame:
    r = ev.policies[policy_id]
    rows = []
    for est in ESTIMATORS:
        e = r.estimates[est]
        row = {
            "estimator": ESTIMATOR_NAMES[est],
            "estimate": _fmt(e.estimate),
            "95% interval": f"[{_fmt(e.ci_low)}, {_fmt(e.ci_high)}]" if e.available else "needs a built-in policy",
            "ESS": "all rows" if e.ess is None else f"{e.ess:.0f}",
        }
        if truth is not None:
            row["gap to truth"] = _fmt(e.estimate - truth) if e.available else "n/a"
        rows.append(row)
    return pd.DataFrame(rows)


def _print_warnings(diag: dict) -> None:
    for w in diag["warnings"]:
        print(f"  ! {w['message']}")


def cmd_simulate(args: argparse.Namespace) -> int:
    cfg = replace(WorldConfig(), n_slots=args.slots)
    df = simulate_log(cfg, args.queries, args.exploration, args.seed)
    write_log_csv(df, args.out)
    print(f"Wrote {len(df)} simulated queries to {args.out} (seed {args.seed}, exploration {args.exploration}).")
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    df = read_log_csv(args.log)
    cfg = WorldConfig()
    try:
        conn = sqlstore.load_log(df, expect_ctx=cfg.d_ctx, expect_item=cfg.d_item)
    except LogFormatError as err:
        print(f"Could not read {args.log}: {err}", file=sys.stderr)
        return 2
    log = sqlstore.log_data(conn)
    summary = sqlstore.log_summary(conn)
    table = sqlstore.per_position_clicks(conn)
    conn.close()
    cfg = replace(cfg, n_slots=log.n_slots)
    ids = list(POLICY_IDS) if args.policy == "all" else [args.policy]
    try:
        ev = evaluate_log(
            log, ids, cfg=cfg, exploration=args.exploration, clip=args.clip,
            n_boot=args.bootstrap, seed=args.seed, per_position=table,
        )
    except LogFormatError as err:
        print(str(err), file=sys.stderr)
        return 2

    print(f"Log: {summary['n_queries']:.0f} queries, {summary['mean_clicks']:.4f} clicks per query under the logging policy.")
    print("Clicks by position:")
    print(table.to_string(index=False))
    for pid in ids:
        print(f"\n{ev.policies[pid].name}  (clip {args.clip if args.clip else 'off'})")
        print(_estimate_rows(ev, pid).to_string(index=False))
        _print_warnings(ev.policies[pid].diagnostics)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(ev.to_dict(), fh, indent=2, allow_nan=True)
        print(f"\nWrote results to {args.json}")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    cfg = replace(WorldConfig(), n_slots=args.slots)
    cmp = run_comparison(
        cfg, n_queries=args.queries, exploration=args.exploration, seed=args.seed,
        clip=args.clip, n_boot=args.bootstrap, n_online=args.online,
    )
    ev = cmp.evaluation
    print(f"Simulated log: {ev.n} queries, exploration {args.exploration}, seed {args.seed}. "
          f"Logged clicks per query: {ev.logged_mean:.4f}")
    for pid in POLICY_IDS:
        truth = cmp.truth[pid]["value"]
        print(f"\n{ev.policies[pid].name}: true online value {truth:.4f}")
        print(_estimate_rows(ev, pid, truth).to_string(index=False))
        _print_warnings(ev.policies[pid].diagnostics)
    t = cmp.table()
    print("\nTrue order:  " + " > ".join(t["truth"].sort_values(ascending=False).index))
    for est, bad in cmp.rank_agreement().items():
        order = " > ".join(t[est].sort_values(ascending=False, na_position="last").index)
        print(f"{ESTIMATOR_NAMES[est]:<26} {bad} pairs out of order   {order}")
    if args.json:
        out = ev.to_dict() | {"truth": cmp.truth, "rank_agreement": cmp.rank_agreement()}
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=2)
        print(f"\nWrote results to {args.json}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="betterrankings",
        description="Estimate how a new ranking policy would perform from logged data alone.",
    )
    sub = p.add_subparsers(dest="command", required=True)

    def sim_args(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--queries", type=int, default=DEFAULTS["n_queries"], help="number of logged queries")
        sp.add_argument("--exploration", type=float, default=DEFAULTS["exploration"],
                        help="share of rankings the logging policy picks uniformly at random (0 to 1)")
        sp.add_argument("--seed", type=int, default=DEFAULTS["seed"], help="run seed")
        sp.add_argument("--slots", type=int, default=WorldConfig().n_slots, help="items per list (2 to 6)")

    s = sub.add_parser("simulate", help="write a simulated log CSV")
    sim_args(s)
    s.add_argument("--out", default="log.csv", help="output CSV path")
    s.set_defaults(func=cmd_simulate)

    policies = ", ".join(f"{q.id} ({q.name})" for q in POLICIES)
    e = sub.add_parser("evaluate", help="estimate a policy's value from a log CSV")
    e.add_argument("--log", required=True, help="log CSV (see README for the columns)")
    e.add_argument("--policy", default="relevance",
                   help=f"one of: {policies}, {TARGET_COLUMN} (use that CSV column), or all")
    e.add_argument("--exploration", type=float, default=DEFAULTS["exploration"],
                   help="logging exploration assumed by the current and noisy_logging policies")
    e.add_argument("--clip", type=_clip, default=DEFAULTS["clip"], help="weight clip level, or 'none'")
    e.add_argument("--bootstrap", type=int, default=DEFAULTS["n_boot"], help="bootstrap resamples")
    e.add_argument("--seed", type=int, default=0, help="seed for cross-fitting folds and the bootstrap")
    e.add_argument("--json", help="also write results to this JSON file")
    e.set_defaults(func=cmd_evaluate)

    c = sub.add_parser("compare", help="simulate, estimate every policy, and check against a true online test")
    sim_args(c)
    c.add_argument("--clip", type=_clip, default=DEFAULTS["clip"], help="weight clip level, or 'none'")
    c.add_argument("--bootstrap", type=int, default=DEFAULTS["n_boot"], help="bootstrap resamples")
    c.add_argument("--online", type=int, default=DEFAULTS["n_online"], help="queries in the simulated online test")
    c.add_argument("--json", help="also write results to this JSON file")
    c.set_defaults(func=cmd_compare)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "policy", None) not in (None, "all", TARGET_COLUMN, *POLICY_IDS):
        print(f"Unknown policy {args.policy!r}. Choose one of: {', '.join(POLICY_IDS)}, "
              f"{TARGET_COLUMN}, all", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
