import itertools

import numpy as np
import pytest

from betterrankings.estimators import bootstrap_indices, doubly_robust, ips
from betterrankings.logs import simulate_log
from betterrankings.perms import perm_index, permutations
from betterrankings.policies import POLICY_IDS, expected_clicks, policy_probs, plackett_luce_probs
from betterrankings.rng import Rng, derive_seed
from betterrankings import sqlstore
from betterrankings.world import WorldConfig, make_params, true_click_matrix

CFG = WorldConfig()


def _mulberry32_scalar(seed: int, n: int) -> list[float]:
    """Textbook sequential mulberry32, to check the vectorised version."""
    a, out = seed & 0xFFFFFFFF, []
    for _ in range(n):
        a = (a + 0x6D2B79F5) & 0xFFFFFFFF
        t = ((a ^ (a >> 15)) * (a | 1)) & 0xFFFFFFFF
        t = ((t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF) ^ t
        out.append(((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0)
    return out


def test_rng_matches_sequential_mulberry32():
    r = Rng(12345)
    first = np.concatenate([r.uniforms(3), r.uniforms(5)])
    assert first.tolist() == _mulberry32_scalar(12345, 8)
    assert derive_seed(1, "a") != derive_seed(1, "b") != derive_seed(2, "a")
    z = Rng(3, "normals").normals(20000)
    assert abs(z.mean()) < 0.03 and abs(z.std() - 1) < 0.03


def test_permutation_index_round_trip():
    for m in range(1, 6):
        perms = permutations(m)
        assert perms.tolist() == [list(p) for p in itertools.permutations(range(m))]
        assert perm_index(perms).tolist() == list(range(len(perms)))


def _log_features(n=300, seed=3):
    df = simulate_log(CFG, n, 0.2, seed)
    conn = sqlstore.load_log(df)
    return df, sqlstore.log_data(conn)


@pytest.mark.parametrize("pid", POLICY_IDS)
def test_policy_probabilities_sum_to_one(pid):
    _, log = _log_features()
    p = policy_probs(pid, log.features, CFG, make_params(CFG), 0.2)
    assert p.shape == (len(log), 24)
    assert np.allclose(p.sum(axis=1), 1.0)
    assert (p >= 0).all()


def test_plackett_luce_is_order_equivariant():
    s = np.array([[0.3, -1.2, 2.0, 0.1]])
    perm = np.array([2, 0, 3, 1])
    p = plackett_luce_probs(s, 0.7)[0]
    q = plackett_luce_probs(s[:, perm], 0.7)[0]
    # A ranking of the original slots, re-expressed in the shuffled slot order.
    inv = np.argsort(perm)
    for a, ranking in enumerate(permutations(4)):
        assert p[a] == pytest.approx(q[perm_index(inv[ranking][None])[0]])


def test_logged_propensity_matches_current_policy():
    _, log = _log_features()
    p = policy_probs("current", log.features, CFG, make_params(CFG), 0.2)
    # Columns follow the shown order, so ranking 0 is the one that was logged.
    assert np.allclose(p[:, 0], log.propensity, rtol=1e-12)


def test_position_bias_in_clicks():
    df = simulate_log(CFG, 4000, 1.0, 9)  # uniform logging: position is independent of item
    rates = [df[f"click_{k}"].mean() for k in range(1, 5)]
    assert rates == sorted(rates, reverse=True)
    assert rates[0] > 2.5 * rates[3]


def test_dr_with_true_model_survives_wrong_propensities():
    """Doubly robust stays on target when the reward model is right but the logged weights are not."""
    df, log = _log_features(n=4000, seed=21)
    params = make_params(CFG)
    g = true_click_matrix(CFG, params, log.features)
    probs = policy_probs("relevance_soft", log.features, CFG, params, 0.2)
    truth_here = float(expected_clicks(probs, g).mean())  # true value on these very queries
    q_logged = g[:, np.arange(4), np.arange(4)].sum(axis=1)
    dm = expected_clicks(probs, g)
    noise = np.exp(Rng(5, "noise").normals(len(log)) * 0.8)
    bad_w = probs[:, 0] / (log.propensity * noise)
    idx = bootstrap_indices(len(log), 200, 1)
    dr = doubly_robust(log.rewards, bad_w, dm, q_logged, None, idx)
    wrong_ips = ips(log.rewards, bad_w, idx)
    assert dr.ci_low <= truth_here <= dr.ci_high
    assert not wrong_ips.ci_low <= truth_here <= wrong_ips.ci_high
