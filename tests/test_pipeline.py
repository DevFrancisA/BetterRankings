import json

import numpy as np
import pandas as pd
import pytest

from betterrankings import sqlstore
from betterrankings.cli import main
from betterrankings.evaluate import evaluate_frame, run_comparison
from betterrankings.logs import LogFormatError, read_log_csv, simulate_log, write_log_csv
from betterrankings.reward_model import cross_fit
from betterrankings.world import WorldConfig

CFG = WorldConfig()


@pytest.fixture(scope="module")
def log_df():
    return simulate_log(CFG, 800, 0.3, 4)


def test_sql_aggregations_match_pandas(log_df):
    conn = sqlstore.load_log(log_df)
    table = sqlstore.per_position_clicks(conn)
    for k in range(1, 5):
        row = table[table["position"] == k].iloc[0]
        assert row["clicks"] == log_df[f"click_{k}"].sum()
        assert row["click_rate"] == pytest.approx(log_df[f"click_{k}"].mean())
    log = sqlstore.log_data(conn)
    expect = log_df[[f"click_{k}" for k in range(1, 5)]].sum(axis=1).to_numpy()
    assert np.array_equal(log.rewards, expect)
    summary = sqlstore.log_summary(conn)
    assert summary["n_queries"] == 800
    assert summary["mean_clicks"] == pytest.approx(expect.mean())
    assert len(sqlstore.item_click_rates(conn, min_impressions=1, limit=5)) == 5


def test_csv_round_trip_is_exact(log_df, tmp_path):
    path = tmp_path / "log.csv"
    write_log_csv(log_df, path)
    back = read_log_csv(path)
    for col in log_df.columns:
        if col == "query_id":
            assert back[col].tolist() == log_df[col].astype(str).tolist()
        else:
            assert np.array_equal(back[col].to_numpy(), log_df[col].to_numpy()), col


def test_bad_logs_are_rejected(log_df):
    with pytest.raises(LogFormatError, match="propensity"):
        sqlstore.load_log(log_df.assign(propensity=0.0))
    with pytest.raises(LogFormatError, match="click"):
        sqlstore.load_log(log_df.assign(click_1=2))
    with pytest.raises(LogFormatError, match="missing"):
        sqlstore.load_log(log_df.drop(columns=["item_2_f1"]))


def test_cross_fit_never_scores_its_own_training_rows(log_df):
    log = sqlstore.log_data(sqlstore.load_log(log_df))
    fit = cross_fit(log.features, log.clicks, 2, seed=8)
    in_fold0 = fit.folds == 0
    flipped = log.clicks.copy()
    flipped[in_fold0] = 1 - flipped[in_fold0]
    refit = cross_fit(log.features, flipped, 2, seed=8)
    # Changing fold 0's clicks must not move fold 0's predictions...
    assert np.array_equal(fit.q[in_fold0], refit.q[in_fold0])
    # ...but does move the other fold, whose model trained on them.
    assert not np.allclose(fit.q[~in_fold0], refit.q[~in_fold0])


def test_runs_are_reproducible_from_a_seed():
    a = run_comparison(CFG, n_queries=600, seed=13, n_boot=50, n_online=2000)
    b = run_comparison(CFG, n_queries=600, seed=13, n_boot=50, n_online=2000)
    c = run_comparison(CFG, n_queries=600, seed=14, n_boot=50, n_online=2000)
    pd.testing.assert_frame_equal(a.log, b.log)
    assert a.evaluation.to_dict() == b.evaluation.to_dict()
    assert a.truth == b.truth
    assert not a.log.equals(c.log)


def test_on_policy_estimates_match_the_logged_mean(log_df):
    ev = evaluate_frame(log_df, ["current"], exploration=0.3, n_boot=20)
    est = ev.policies["current"].estimates
    assert est["ips"].estimate == pytest.approx(ev.logged_mean, rel=1e-9)
    assert est["snips"].estimate == pytest.approx(ev.logged_mean, rel=1e-9)
    assert ev.policies["current"].diagnostics["warnings"] == []


def test_estimates_land_near_the_truth_with_good_overlap():
    """Averaged over a few independent logs, the weighting estimators track the online test."""
    pids = ["relevance_soft", "noisy_logging"]
    errors = {(pid, est): [] for pid in pids for est in ("ips", "snips", "doubly_robust")}
    for seed in (31, 32, 33, 34):
        cmp = run_comparison(CFG, n_queries=6000, exploration=0.5, seed=seed, n_boot=20, n_online=20000,
                             policy_ids=pids)
        for pid, est in errors:
            errors[pid, est].append(cmp.evaluation.policies[pid].estimates[est].estimate - cmp.truth[pid]["value"])
    for key, errs in errors.items():
        assert abs(np.mean(errs)) < 0.03, key


def test_target_propensity_column(log_df):
    df = log_df.assign(target_propensity=log_df["propensity"])
    ev = evaluate_frame(df, ["target_propensity"], n_boot=20)
    est = ev.policies["target_propensity"].estimates
    assert est["ips"].estimate == pytest.approx(ev.logged_mean)
    assert not est["doubly_robust"].available


def test_cli_simulate_evaluate_compare(tmp_path, capsys):
    log = tmp_path / "log.csv"
    assert main(["simulate", "--queries", "300", "--seed", "2", "--out", str(log)]) == 0
    first = log.read_bytes()
    assert main(["simulate", "--queries", "300", "--seed", "2", "--out", str(log)]) == 0
    assert log.read_bytes() == first
    out = tmp_path / "res.json"
    assert main(["evaluate", "--log", str(log), "--policy", "all", "--bootstrap", "20", "--json", str(out)]) == 0
    assert set(json.loads(out.read_text())["policies"]) >= {"relevance", "popularity"}
    assert main(["compare", "--queries", "300", "--bootstrap", "20", "--online", "1000"]) == 0
    assert "True order" in capsys.readouterr().out
    assert main(["evaluate", "--log", str(log), "--policy", "nope"]) == 2
