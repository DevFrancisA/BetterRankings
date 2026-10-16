"""The Open Bandit Dataset pipeline, on a 3,000-row public sample of each log."""

from pathlib import Path

import numpy as np
import pytest

from betterrankings import obd
from betterrankings.cli import main

DATA = Path(__file__).parent / "data" / "obd"


@pytest.fixture(scope="module")
def logs():
    return obd.read_log(DATA, "random", "all"), obd.read_log(DATA, "bts", "all")


def test_reads_the_log(logs):
    rnd, bts = logs
    assert len(rnd) == 3000 and rnd.n_items == 80
    assert set(np.unique(rnd.position)) == {0, 1, 2}
    assert np.allclose(rnd.pscore, 1 / 80)
    # rows are grouped into recommendations by time, with contiguous ids
    ids = np.unique(rnd.impression)
    assert ids[0] == 0 and ids[-1] == len(ids) - 1 and len(ids) <= 3000
    assert rnd.hour.min() >= 0 and rnd.hour.max() < 7 * 24 and rnd.user.shape == (3000, 4)


def test_cache_round_trip(tmp_path):
    a = obd.read_log(DATA, "bts", "all", cache_dir=tmp_path)
    b = obd.read_log(DATA, "bts", "all", cache_dir=tmp_path)
    assert (tmp_path / "obd_bts_all.npz").exists()
    for field in obd.FIELDS:
        assert np.array_equal(getattr(a, field), getattr(b, field)), field


def test_policies_are_distributions(logs):
    _, bts = logs
    for pi in (obd.empirical_policy(bts), obd.uniform_policy(80)):
        assert np.allclose(pi.sum(axis=0), 1.0)
    hourly = obd.hourly_policy(bts)
    assert hourly.shape == (168, 80, 3) and np.allclose(hourly.sum(axis=1), 1.0)
    prior = obd.prior_policy(np.full(5, 2.0), np.full(5, 5.0), n_sim=20_000)
    assert np.allclose(prior.sum(axis=0), 1.0)


def test_on_policy_estimates_equal_the_logged_rate(logs):
    rnd, _ = logs
    est, diag = obd.estimate_policy(rnd, obd.uniform_policy(80), clip=None, n_boot=30)
    ctr = rnd.click.mean()
    assert est["ips"].estimate == pytest.approx(ctr)
    assert est["snips"].estimate == pytest.approx(ctr)
    assert diag["warnings"] == []


def test_cross_fit_scores_rows_out_of_fold(logs):
    rnd, bts = logs
    q, d = obd.cross_fit_terms(rnd, obd.empirical_policy(bts), seed=1)
    assert q.shape == d.shape == (3000,)
    assert ((q > 0) & (q < 1)).all() and ((d > 0) & (d < 1)).all()


def test_heavy_weights_trigger_overlap_warning(logs):
    _, bts = logs
    _, diag = obd.estimate_policy(bts, obd.uniform_policy(80), clip=None, n_boot=10)
    assert diag["max_weight"] > 10


def test_cli_runs_on_the_sample(tmp_path, capsys):
    out = tmp_path / "obd.json"
    assert main(["obd", "--data", str(DATA), "--bootstrap", "20", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "bts_weekly_from_random" in text and "random_from_bts" in text
    assert out.exists()
    assert main(["obd", "--data", str(tmp_path / "missing")]) == 2
