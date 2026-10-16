import numpy as np
import pytest

from betterrankings.estimators import (
    bootstrap_indices,
    doubly_robust,
    effective_sample_size,
    ips,
    naive,
    overlap_diagnostics,
    percentile_ci,
    snips,
    clipped_ips,
    reward_model,
)

N_LOGS, N_ROWS = 600, 400


def _row_means(terms: np.ndarray) -> tuple[float, float]:
    """Average of per-log estimates and its Monte Carlo standard error."""
    return float(terms.mean()), float(terms.std(ddof=1) / np.sqrt(len(terms)))


def _apply(tiny, est):
    a, r = tiny.draw(N_LOGS, N_ROWS, seed=1)
    idx = np.zeros((1, N_ROWS), dtype=np.int64)  # bootstrap not needed here
    return np.array([est(a[i], r[i], idx).estimate for i in range(N_LOGS)])


def test_ips_is_unbiased_on_known_answer(tiny_world):
    t = tiny_world
    values = _apply(t, lambda a, r, idx: ips(r, t.pi[a] / t.mu[a], idx))
    mean, se = _row_means(values)
    assert abs(mean - t.truth) < 4 * se


def test_clip_above_max_weight_is_plain_ips(tiny_world):
    t = tiny_world
    big = float((t.pi / t.mu).max()) + 1
    a, r = t.draw(1, N_ROWS, seed=2)
    w = t.pi[a[0]] / t.mu[a[0]]
    idx = bootstrap_indices(N_ROWS, 50, 3)
    assert clipped_ips(r[0], w, big, idx).estimate == pytest.approx(ips(r[0], w, idx).estimate)


def test_snips_is_close_and_naive_is_biased(tiny_world):
    t = tiny_world
    sn = _apply(t, lambda a, r, idx: snips(r, t.pi[a] / t.mu[a], idx))
    nv = _apply(t, lambda a, r, idx: naive(r, t.pi[a], idx))
    sn_mean, sn_se = _row_means(sn)
    nv_mean, nv_se = _row_means(nv)
    # SNIPS has O(1/n) bias, small next to its spread at n=400.
    assert abs(sn_mean - t.truth) < 4 * sn_se + 0.01
    # Ignoring the logging propensities over-weights rankings the old policy liked.
    assert abs(nv_mean - t.truth) > 10 * nv_se


def test_reward_model_with_true_values_is_unbiased(tiny_world):
    t = tiny_world
    dm_true = np.full(N_ROWS, t.truth)
    values = _apply(t, lambda a, r, idx: reward_model(dm_true, idx))
    assert values.mean() == pytest.approx(t.truth)


def test_dr_holds_up_when_reward_model_is_wrong(tiny_world):
    t = tiny_world
    wrong_q = np.full(len(t.values), 2.0)  # wildly off for every ranking
    wrong_dm = float(t.pi @ wrong_q)

    def est(a, r, idx):
        dm = np.full(len(a), wrong_dm)
        return doubly_robust(r, t.pi[a] / t.mu[a], dm, wrong_q[a], None, idx)

    values = _apply(t, est)
    mean, se = _row_means(values)
    assert abs(mean - t.truth) < 4 * se
    assert abs(wrong_dm - t.truth) > 1.0  # the model alone is far off


def test_dr_holds_up_when_weights_are_wrong(tiny_world):
    t = tiny_world
    bad_mu = t.mu * np.array([2.5, 0.4, 1.7, 0.6, 3.0, 0.3])  # mis-recorded propensities
    dm = np.full(N_ROWS, t.truth)  # correct reward model

    def dr(a, r, idx):
        return doubly_robust(r, t.pi[a] / bad_mu[a], dm, t.values[a], None, idx)

    def wrong_ips(a, r, idx):
        return ips(r, t.pi[a] / bad_mu[a], idx)

    dr_mean, dr_se = _row_means(_apply(t, dr))
    ips_mean, ips_se = _row_means(_apply(t, wrong_ips))
    assert abs(dr_mean - t.truth) < 4 * dr_se
    assert abs(ips_mean - t.truth) > 10 * ips_se


def test_clipping_lowers_variance_on_heavy_tailed_weights():
    rng = np.random.default_rng(0)
    n_logs, n = 400, 300
    # Lognormal weights with sigma 2: a few rows carry most of the weight.
    w = rng.lognormal(mean=-2.0, sigma=2.0, size=(n_logs, n))
    r = rng.binomial(1, 0.3, size=(n_logs, n)).astype(float)
    idx = np.zeros((1, n), dtype=np.int64)
    raw = np.array([ips(r[i], w[i], idx).estimate for i in range(n_logs)])
    cut = np.array([clipped_ips(r[i], w[i], 5.0, idx).estimate for i in range(n_logs)])
    assert np.percentile(w, 99.9) > 50  # really heavy tailed
    assert cut.var() < 0.25 * raw.var()


def test_effective_sample_size():
    assert effective_sample_size(np.ones(50)) == pytest.approx(50)
    assert effective_sample_size(np.array([1.0, 0, 0, 0])) == pytest.approx(1)
    assert effective_sample_size(np.zeros(5)) == 0.0


def test_percentile_ci_matches_numpy():
    b = np.random.default_rng(4).normal(size=201)
    lo, hi = percentile_ci(b)
    assert lo == pytest.approx(np.quantile(b, 0.025))
    assert hi == pytest.approx(np.quantile(b, 0.975))
    assert np.isnan(percentile_ci(np.array([np.nan]))[0])


def test_bootstrap_interval_covers_the_point():
    rng = np.random.default_rng(5)
    r = rng.binomial(1, 0.4, 500).astype(float)
    e = ips(r, np.ones(500), bootstrap_indices(500, 300, 11))
    assert e.ci_low < e.estimate < e.ci_high
    assert e.ci_high - e.ci_low < 0.2


def test_overlap_warnings():
    good = overlap_diagnostics(np.ones(1000))
    assert good["warnings"] == []
    sparse = np.zeros(1000)
    sparse[:5] = 200.0
    codes = {w["code"] for w in overlap_diagnostics(sparse)["warnings"]}
    assert "poor_overlap" in codes and "extreme_weight" in codes
    assert overlap_diagnostics(np.zeros(10))["warnings"][0]["code"] == "no_support"
    off = {w["code"] for w in overlap_diagnostics(np.full(1000, 2.0))["warnings"]}
    assert off == {"weights_off"}
