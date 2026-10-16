"""Off-policy estimators of a target policy's expected clicks per query.

Notation, for logged query i with total clicks r_i:

    mu_i  logging propensity of the shown ranking
    pi_i  target policy's probability of that same ranking
    w_i   importance weight pi_i / mu_i
    q_i   reward model's predicted clicks for the shown ranking
    d_i   reward model's expected clicks under the target policy

Every estimator is written as per-row terms, so the bootstrap only resamples
rows. The reward model is not refitted inside the bootstrap; its intervals
therefore leave out model-fitting uncertainty.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .rng import Rng

ESTIMATORS: tuple[str, ...] = ("naive", "ips", "snips", "clipped_ips", "reward_model", "doubly_robust")
ESTIMATOR_NAMES = {
    "naive": "Naive click rate",
    "ips": "Inverse propensity (IPS)",
    "snips": "Self-normalised IPS",
    "clipped_ips": "Clipped IPS",
    "reward_model": "Reward model",
    "doubly_robust": "Doubly robust",
}
CI_LEVEL = 0.95


@dataclass
class Estimate:
    estimate: float
    ci_low: float
    ci_high: float
    ess: float | None  # effective sample size of the weights this estimator uses
    n: int
    available: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


def bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    u = Rng(seed, "bootstrap").uniforms(n_boot * n).reshape(n_boot, n)
    return np.floor(u * n).astype(np.int64)


def quantile_sorted(b: np.ndarray, q: float) -> float:
    """Linear-interpolation quantile of an already sorted array (NumPy's default)."""
    h = (len(b) - 1) * q
    lo = int(np.floor(h))
    hi = min(lo + 1, len(b) - 1)
    return float(b[lo] + (h - lo) * (b[hi] - b[lo]))


def percentile_ci(boots: np.ndarray, level: float = CI_LEVEL) -> tuple[float, float]:
    b = np.sort(boots[np.isfinite(boots)])
    if len(b) == 0:
        return float("nan"), float("nan")
    tail = (1.0 - level) / 2.0
    return quantile_sorted(b, tail), quantile_sorted(b, 1.0 - tail)


def effective_sample_size(w: np.ndarray) -> float:
    """(sum w)^2 / sum w^2: how many equally weighted rows the weights are worth."""
    sq = float(np.dot(w, w))
    return 0.0 if sq == 0.0 else float(w.sum()) ** 2 / sq


def clip_weights(w: np.ndarray, clip: float | None) -> np.ndarray:
    return w if clip is None or not np.isfinite(clip) else np.minimum(w, clip)


def _mean_estimate(t: np.ndarray, idx: np.ndarray, ess: float | None) -> Estimate:
    lo, hi = percentile_ci(t[idx].mean(axis=1))
    return Estimate(float(t.mean()), lo, hi, ess, len(t))


def _ratio_estimate(a: np.ndarray, b: np.ndarray, idx: np.ndarray, ess: float | None) -> Estimate:
    total = float(b.sum())
    point = float(a.sum()) / total if total > 0 else float("nan")
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = a[idx].sum(axis=1) / b[idx].sum(axis=1)
    lo, hi = percentile_ci(boots)
    return Estimate(point, lo, hi, ess, len(a))


def _unavailable(n: int) -> Estimate:
    nan = float("nan")
    return Estimate(nan, nan, nan, None, n, available=False)


def naive(rewards: np.ndarray, pi: np.ndarray, idx: np.ndarray) -> Estimate:
    """Average clicks on logged queries where the target would have shown the same list.

    Rows are weighted by the target's probability of the shown ranking but the
    logging propensity is ignored, so lists the old policy liked to show (and the
    contexts where it showed them) are over-represented.
    """
    return _ratio_estimate(pi * rewards, pi, idx, effective_sample_size(pi))


def ips(rewards: np.ndarray, w: np.ndarray, idx: np.ndarray) -> Estimate:
    return _mean_estimate(w * rewards, idx, effective_sample_size(w))


def snips(rewards: np.ndarray, w: np.ndarray, idx: np.ndarray) -> Estimate:
    return _ratio_estimate(w * rewards, w, idx, effective_sample_size(w))


def clipped_ips(rewards: np.ndarray, w: np.ndarray, clip: float | None, idx: np.ndarray) -> Estimate:
    wc = clip_weights(w, clip)
    return _mean_estimate(wc * rewards, idx, effective_sample_size(wc))


def reward_model(dm: np.ndarray, idx: np.ndarray) -> Estimate:
    return _mean_estimate(dm, idx, None)


def doubly_robust(
    rewards: np.ndarray, w: np.ndarray, dm: np.ndarray, q_logged: np.ndarray, clip: float | None, idx: np.ndarray
) -> Estimate:
    """d_i + min(w_i, c) * (r_i - q_i): the model's guess, corrected by weighted residuals."""
    wc = clip_weights(w, clip)
    return _mean_estimate(dm + wc * (rewards - q_logged), idx, effective_sample_size(wc))


def all_estimates(
    rewards: np.ndarray,
    propensity: np.ndarray,
    pi_logged: np.ndarray,
    dm: np.ndarray | None,
    q_logged: np.ndarray | None,
    clip: float | None,
    idx: np.ndarray,
) -> dict[str, Estimate]:
    w = pi_logged / propensity
    n = len(rewards)
    out = {
        "naive": naive(rewards, pi_logged, idx),
        "ips": ips(rewards, w, idx),
        "snips": snips(rewards, w, idx),
        "clipped_ips": clipped_ips(rewards, w, clip, idx),
    }
    if dm is None or q_logged is None:
        out["reward_model"] = _unavailable(n)
        out["doubly_robust"] = _unavailable(n)
    else:
        out["reward_model"] = reward_model(dm, idx)
        out["doubly_robust"] = doubly_robust(rewards, w, dm, q_logged, clip, idx)
    return out


# Overlap diagnostics ---------------------------------------------------------

POOR_ESS_FRACTION = 0.01
POOR_ESS_MIN = 30.0
LIMITED_ESS_FRACTION = 0.10
EXTREME_WEIGHT_SHARE = 0.10
MEAN_WEIGHT_TOLERANCE = 0.30


def overlap_diagnostics(w: np.ndarray, clip: float | None = None) -> dict:
    """Weight summaries plus warnings, each with a code and a plain-language message."""
    n = len(w)
    total = float(w.sum())
    ess = effective_sample_size(w)
    max_w = float(w.max()) if n else 0.0
    share = max_w / total if total > 0 else float("nan")
    mean_w = total / n if n else float("nan")
    clipped = int((w > clip).sum()) if clip is not None and np.isfinite(clip) else 0
    warnings: list[dict] = []
    if total == 0:
        warnings.append({
            "code": "no_support",
            "level": "warn",
            "message": "The target policy never picks any ranking that appears in the log, "
            "so weighting estimators have nothing to work with.",
        })
    else:
        if ess < max(POOR_ESS_MIN, POOR_ESS_FRACTION * n):
            warnings.append({
                "code": "poor_overlap",
                "level": "warn",
                "message": f"Poor overlap: the weights are worth about {ess:.0f} of {n} logged queries. "
                "The logging policy rarely showed the lists this policy would show, so estimates are unreliable.",
            })
        elif ess < LIMITED_ESS_FRACTION * n:
            warnings.append({
                "code": "limited_overlap",
                "level": "info",
                "message": f"Limited overlap: the weights are worth about {ess:.0f} of {n} logged queries. "
                "Expect wide intervals.",
            })
        if share > EXTREME_WEIGHT_SHARE:
            warnings.append({
                "code": "extreme_weight",
                "level": "warn",
                "message": f"One logged query carries {share:.0%} of the total weight, "
                "so a single click can swing the weighted estimates.",
            })
        if abs(mean_w - 1.0) > MEAN_WEIGHT_TOLERANCE:
            warnings.append({
                "code": "weights_off",
                "level": "warn",
                "message": f"The weights average {mean_w:.2f} instead of about 1. Either the log rarely "
                "covers this policy's choices or the logged propensities are off.",
            })
    return {
        "n": n,
        "ess": ess,
        "ess_fraction": ess / n if n else float("nan"),
        "max_weight": max_w,
        "max_weight_share": share,
        "mean_weight": mean_w,
        "fraction_nonzero": float((w > 0).mean()) if n else float("nan"),
        "n_clipped": clipped,
        "warnings": warnings,
    }
