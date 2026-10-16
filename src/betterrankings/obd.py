"""Real-data check on the Open Bandit Dataset (Saito et al., 2020).

ZOZOTOWN ran two recommendation policies side by side on live traffic for a
week: a uniform random policy and Bernoulli Thompson sampling (BTS). Each
impression shows 3 items (left, center, right) out of 34 to 80, and every row of
the logs records the item shown at one position, whether it was clicked, and the
logging policy's probability of that choice. Because both policies were logged,
each one's real click rate is known, which gives off-policy estimates a real
answer to be checked against:

* ``bts_from_random``: estimate BTS's click rate from the random policy's logs.
  BTS's choice probabilities are taken as its empirical average behaviour, how
  often each item appeared at each position in BTS's own 4M to 12M rows. BTS's
  choices barely depend on the user features, so this is close to non-contextual.
* ``random_from_bts``: estimate the random policy's click rate from BTS's logs.
  Here every quantity is exact: the logged propensities are BTS's own and the
  target policy is uniform. Its weights are heavy where BTS rarely showed an item.

The estimand is the click rate per impression slot, the same quantity as the
logged policy's mean click. Estimators work position by position: the weight of
a row is target(item | position) / logging(item | position).

The data is CC BY 4.0. Cite: Yuta Saito, Shunsuke Aihara, Megumi Matsutani,
Yusuke Narita. "Open Bandit Dataset and Pipeline: Towards Realistic and
Reproducible Off-Policy Evaluation", NeurIPS Datasets and Benchmarks, 2021.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .estimators import (
    ESTIMATORS,
    Estimate,
    clip_weights,
    effective_sample_size,
    overlap_diagnostics,
    percentile_ci,
)
from .reward_model import fit_logistic
from .rng import Rng
from .world import sigmoid

CAMPAIGNS = ("all", "men", "women")
POLICIES = ("random", "bts")
FIELDS = ("item", "position", "click", "pscore", "user", "affinity", "impression", "hour")
START_DAY = 24  # the test ran 2019-11-24 to 2019-11-30
N_USER_FEATURES = 4
MAX_TRAIN_ROWS = 1_000_000
_CHUNK = 1_000_000


@dataclass
class ObdLog:
    policy: str
    campaign: str
    item: np.ndarray  # (n,) int16
    position: np.ndarray  # (n,) int8, 0-based
    click: np.ndarray  # (n,) int8
    pscore: np.ndarray  # (n,) float64, logging probability of the item at that position
    user: np.ndarray  # (n, 4) int16 category codes
    affinity: np.ndarray  # (n, n_items) int8 user-item affinity
    impression: np.ndarray  # (n,) int64, rows of one recommendation share an id
    hour: np.ndarray  # (n,) int16, hours since the start of the test (2019-11-24 00:00 UTC)

    def __len__(self) -> int:
        return len(self.click)

    @property
    def n_items(self) -> int:
        return self.affinity.shape[1]

    def take(self, rows: np.ndarray) -> "ObdLog":
        return ObdLog(
            self.policy,
            self.campaign,
            self.item[rows],
            self.position[rows],
            self.click[rows],
            self.pscore[rows],
            self.user[rows],
            self.affinity[rows],
            self.impression[rows],
            self.hour[rows],
        )


IMPRESSION_GAP_SECONDS = 0.02


def _impression_ids(seconds: np.ndarray) -> np.ndarray:
    """Group rows into recommendations: the rows of one recommendation are logged within milliseconds,
    so a new group starts after a gap of more than 20 ms. On the full logs this puts 94% of rows in
    groups of exactly 3 (left, center, right)."""
    order = np.argsort(seconds, kind="stable")
    new = np.concatenate([[0], (np.diff(seconds[order]) > IMPRESSION_GAP_SECONDS).astype(np.int64)])
    ids = np.empty(len(seconds), dtype=np.int64)
    ids[order] = np.cumsum(new)
    return ids


def read_log(root: str | Path, policy: str, campaign: str, cache_dir: str | Path | None = None) -> ObdLog:
    """Load ``{root}/{policy}/{campaign}/{campaign}.csv`` into compact arrays (cached as .npz if asked)."""
    if policy not in POLICIES or campaign not in CAMPAIGNS:
        raise ValueError(f"policy must be one of {POLICIES} and campaign one of {CAMPAIGNS}")
    cache = Path(cache_dir) / f"obd_{policy}_{campaign}.npz" if cache_dir else None
    if cache and cache.exists():
        z = np.load(cache)
        return ObdLog(policy, campaign, *(z[k] for k in FIELDS))

    path = Path(root) / policy / campaign / f"{campaign}.csv"
    header = pd.read_csv(path, nrows=0).columns
    aff_cols = sorted((c for c in header if c.startswith("user-item_affinity_")), key=lambda c: int(c.rsplit("_", 1)[1]))
    user_cols = [f"user_feature_{j}" for j in range(N_USER_FEATURES)]
    dtypes = {"timestamp": str, "item_id": np.int16, "position": np.int8, "click": np.int8, "propensity_score": np.float64}
    dtypes |= {c: str for c in user_cols} | {c: np.float32 for c in aff_cols}
    parts: dict[str, list] = {k: [] for k in ("item", "position", "click", "pscore", "affinity", "hour", "seconds")}
    users: list[pd.DataFrame] = []
    for ch in pd.read_csv(path, usecols=list(dtypes), dtype=dtypes, chunksize=_CHUNK):
        parts["item"].append(ch["item_id"].to_numpy())
        parts["position"].append((ch["position"].to_numpy() - 1).astype(np.int8))
        parts["click"].append(ch["click"].to_numpy())
        parts["pscore"].append(ch["propensity_score"].to_numpy())
        parts["affinity"].append(np.rint(ch[aff_cols].to_numpy()).astype(np.int8))
        ts = ch["timestamp"].str  # "2019-11-24 00:00:01.856987+00:00", fraction sometimes missing
        day = ts.slice(8, 10).astype(np.int64) - START_DAY
        hh = ts.slice(11, 13).astype(np.int64)
        parts["hour"].append((day * 24 + hh).to_numpy(np.int16))
        secs = ts.slice(17).str.split("+").str[0].astype(np.float64)
        parts["seconds"].append((day * 86400 + hh * 3600 + ts.slice(14, 16).astype(np.int64) * 60 + secs).to_numpy())
        users.append(ch[user_cols])
    u = pd.concat(users, ignore_index=True)
    user = np.stack([pd.Categorical(u[c]).codes.astype(np.int16) for c in user_cols], axis=1)
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    log = ObdLog(
        policy, campaign, arrays["item"], arrays["position"], arrays["click"], arrays["pscore"],
        user, arrays["affinity"], _impression_ids(arrays["seconds"]), arrays["hour"],
    )
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{k: getattr(log, k) for k in FIELDS})
    return log


# Target policies ------------------------------------------------------------------

def empirical_policy(log: ObdLog) -> np.ndarray:
    """How often each item appeared at each position in a policy's own log, shape (n_items, 3)."""
    counts = np.zeros((log.n_items, 3))
    np.add.at(counts, (log.item, log.position), 1.0)
    return counts / counts.sum(axis=0, keepdims=True)


def hourly_policy(log: ObdLog, n_hours: int = 7 * 24) -> np.ndarray:
    """Empirical behaviour hour by hour, shape (n_hours, n_items, 3); hours with no rows use the overall average."""
    counts = np.zeros((n_hours, log.n_items, 3))
    np.add.at(counts, (np.clip(log.hour, 0, n_hours - 1), log.item, log.position), 1.0)
    totals = counts.sum(axis=1, keepdims=True)
    overall = empirical_policy(log)[None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, counts / totals, overall)


def target_rows(target: np.ndarray, log: ObdLog) -> np.ndarray:
    """Target probability of every item at each row's position (and hour, for hourly targets), shape (n, n_items)."""
    if target.ndim == 3:
        hour = np.clip(log.hour, 0, target.shape[0] - 1)
        return target[hour[:, None], np.arange(target.shape[1])[None, :], log.position[:, None]]
    return target[:, log.position].T


def uniform_policy(n_items: int) -> np.ndarray:
    return np.full((n_items, 3), 1.0 / n_items)


def read_prior(path: str | Path, campaign: str) -> tuple[np.ndarray, np.ndarray]:
    """Alpha and beta of BTS's published production prior (obp's prior_bts.yaml)."""
    text = Path(path).read_text(encoding="utf-8")
    blocks = re.split(r"^(\w+):\s*$", text, flags=re.M)
    block = dict(zip(blocks[1::2], blocks[2::2]))[campaign]
    grab = lambda key: np.array(  # noqa: E731
        [float(x) for x in re.findall(r"-\s*([\d.]+)", re.search(rf"{key}:\s*\n((?:\s*-\s*[\d.]+\s*\n?)+)", block).group(1))]
    )
    return grab("alpha"), grab("beta")


def prior_policy(alpha: np.ndarray, beta: np.ndarray, n_sim: int = 1_000_000, seed: int = 0) -> np.ndarray:
    """BTS's choice distribution under its published prior: rank items by Beta draws, keep the top 3."""
    rng = np.random.default_rng(seed)
    counts = np.zeros((len(alpha), 3))
    for start in range(0, n_sim, 100_000):
        theta = rng.beta(alpha, beta, size=(min(100_000, n_sim - start), len(alpha)))
        top = np.argsort(-theta, axis=1)[:, :3]
        for k in range(3):
            counts[:, k] += np.bincount(top[:, k], minlength=len(alpha))
    return counts / counts.sum(axis=0, keepdims=True)


# Reward model ------------------------------------------------------------------

@dataclass
class AdditiveClickModel:
    """logit P(click) = b0 + item + position + day + hour of day + user features + affinity * (user's affinity for the item)."""

    intercept: float
    item: np.ndarray  # (n_items,)
    position: np.ndarray  # (3,)
    day: np.ndarray  # (7,)
    hour_of_day: np.ndarray  # (24,)
    user: list[np.ndarray]  # per user feature, one coefficient per category
    affinity: float

    def base(self, log: ObdLog, position: np.ndarray | None = None) -> np.ndarray:
        """Everything except the item terms, per row."""
        pos = log.position if position is None else position
        hour = np.clip(log.hour, 0, 7 * 24 - 1)
        out = self.intercept + self.position[pos] + self.day[hour // 24] + self.hour_of_day[hour % 24]
        for j, coef in enumerate(self.user):
            out = out + coef[np.minimum(log.user[:, j], len(coef) - 1)]
        return out


def _design(log: ObdLog, n_user_levels: list[int]) -> np.ndarray:
    n, k = len(log), log.n_items
    hour = np.clip(log.hour, 0, 7 * 24 - 1)
    cols = [np.ones((n, 1)), np.eye(k)[log.item][:, 1:], np.eye(3)[log.position][:, 1:],
            np.eye(7)[hour // 24][:, 1:], np.eye(24)[hour % 24][:, 1:]]
    for j, levels in enumerate(n_user_levels):
        cols.append(np.eye(levels)[np.minimum(log.user[:, j], levels - 1)][:, 1:])
    cols.append(log.affinity[np.arange(n), log.item].astype(np.float64)[:, None])
    return np.hstack(cols)


def fit_click_model(log: ObdLog, n_user_levels: list[int]) -> AdditiveClickModel:
    beta = fit_logistic(_design(log, n_user_levels), log.click.astype(np.float64))
    k = log.n_items
    at = 1
    item = np.concatenate([[0.0], beta[at : at + k - 1]])
    at += k - 1
    position = np.concatenate([[0.0], beta[at : at + 2]])
    at += 2
    day = np.concatenate([[0.0], beta[at : at + 6]])
    at += 6
    hour_of_day = np.concatenate([[0.0], beta[at : at + 23]])
    at += 23
    user = []
    for levels in n_user_levels:
        user.append(np.concatenate([[0.0], beta[at : at + levels - 1]]))
        at += levels - 1
    return AdditiveClickModel(float(beta[0]), item, position, day, hour_of_day, user, float(beta[at]))


def model_terms(model: AdditiveClickModel, log: ObdLog, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per row: predicted click of the shown item (q) and expected click under the target at that position (d)."""
    n = len(log)
    q = np.empty(n)
    d = np.empty(n)
    for s in range(0, n, 250_000):
        part = log.take(np.arange(s, min(n, s + 250_000)))
        base = model.base(part)
        logits = base[:, None] + model.item[None, :] + model.affinity * part.affinity
        p = sigmoid(logits)
        rows = np.arange(len(part))
        q[s : s + len(part)] = p[rows, part.item]
        d[s : s + len(part)] = (p * target_rows(target, part)).sum(axis=1)
    return q, d


def cross_fit_terms(log: ObdLog, target: np.ndarray, seed: int = 0, n_folds: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Held-out q and d: each impression is scored by a model trained on the other folds only."""
    n_imp = int(log.impression.max()) + 1
    fold = np.floor(Rng(seed, "obd-folds").uniforms(n_imp) * n_folds).astype(np.int64)[log.impression]
    levels = [int(log.user[:, j].max()) + 1 for j in range(N_USER_FEATURES)]
    q = np.empty(len(log))
    d = np.empty(len(log))
    pick = Rng(seed, "obd-train-sample")
    for k in range(n_folds):
        held = np.nonzero(fold == k)[0]
        train = np.nonzero(fold != k)[0]
        if len(train) > MAX_TRAIN_ROWS:  # a uniform sample keeps the fit fast on 12M rows
            u = pick.uniforms(len(train))
            train = train[np.argsort(u)[:MAX_TRAIN_ROWS]]
            train.sort()
        model = fit_click_model(log.take(train), levels)
        q[held], d[held] = model_terms(model, log.take(held), target)
    return q, d


# Estimates with an impression-level bootstrap ----------------------------------------

def _cluster_sums(cluster: np.ndarray, n_clusters: int, *values: np.ndarray) -> list[np.ndarray]:
    return [np.bincount(cluster, weights=v, minlength=n_clusters) for v in values]


def estimate_policy(
    log: ObdLog,
    target: np.ndarray,
    *,
    clip: float | None = 20.0,
    n_boot: int = 200,
    seed: int = 0,
    model: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, Estimate], dict]:
    """All six estimators for ``target`` from ``log``, with impression-clustered bootstrap intervals."""
    r = log.click.astype(np.float64)
    pi = np.empty(len(log))
    for s in range(0, len(log), 500_000):
        part = log.take(np.arange(s, min(len(log), s + 500_000)))
        pi[s : s + len(part)] = target_rows(target, part)[np.arange(len(part)), part.item]
    w = pi / log.pscore
    wc = clip_weights(w, clip)
    q, d = model if model is not None else cross_fit_terms(log, target, seed)
    ones = np.ones(len(log))
    # numerator and denominator per estimator; mean-type estimators divide by the row count
    terms = {
        "naive": (pi * r, pi),
        "ips": (w * r, ones),
        "snips": (w * r, w),
        "clipped_ips": (wc * r, ones),
        "reward_model": (d, ones),
        "doubly_robust": (d + wc * (r - q), ones),
    }
    ess = {"naive": effective_sample_size(pi), "ips": effective_sample_size(w), "snips": effective_sample_size(w),
           "clipped_ips": effective_sample_size(wc), "reward_model": None, "doubly_robust": effective_sample_size(wc)}
    n_clusters = int(log.impression.max()) + 1
    sums = {k: _cluster_sums(log.impression, n_clusters, a, b) for k, (a, b) in terms.items()}
    boots = {k: np.empty(n_boot) for k in terms}
    rng = Rng(seed, "obd-bootstrap")
    for b in range(n_boot):
        counts = np.bincount(np.floor(rng.uniforms(n_clusters) * n_clusters).astype(np.int64), minlength=n_clusters)
        for k, (a_sum, b_sum) in sums.items():
            den = counts @ b_sum
            boots[k][b] = (counts @ a_sum) / den if den > 0 else np.nan
    out = {}
    for k, (a, b) in terms.items():
        total = b.sum()
        lo, hi = percentile_ci(boots[k])
        out[k] = Estimate(float(a.sum() / total) if total > 0 else float("nan"), lo, hi, ess[k], len(log))
    return out, overlap_diagnostics(w, clip)


def clustered_mean(log: ObdLog) -> tuple[float, float]:
    """On-policy click rate and its standard error, treating each recommendation as one unit."""
    n_clusters = int(log.impression.max()) + 1
    c_sum, c_n = _cluster_sums(log.impression, n_clusters, log.click.astype(np.float64), np.ones(len(log)))
    mean = c_sum.sum() / c_n.sum()
    resid = c_sum - mean * c_n
    se = np.sqrt(n_clusters / (n_clusters - 1) * (resid @ resid)) / c_n.sum()
    return float(mean), float(se)


@dataclass
class RealCheck:
    name: str
    campaign: str
    logging_policy: str
    target_policy: str
    n_rows: int
    truth: float
    truth_se: float
    logged_ctr: float
    estimates: dict[str, Estimate]
    diagnostics: dict

    def table(self) -> pd.DataFrame:
        rows = []
        for k in ESTIMATORS:
            e = self.estimates[k]
            rows.append({
                "estimator": k,
                "estimate": e.estimate,
                "ci_low": e.ci_low,
                "ci_high": e.ci_high,
                "relative_error": abs(e.estimate - self.truth) / self.truth,
                "covers_truth": bool(e.ci_low <= self.truth <= e.ci_high),
                "ess": e.ess,
            })
        return pd.DataFrame(rows).set_index("estimator")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "campaign": self.campaign,
            "logging_policy": self.logging_policy,
            "target_policy": self.target_policy,
            "n_rows": self.n_rows,
            "truth": self.truth,
            "truth_se": self.truth_se,
            "logged_ctr": self.logged_ctr,
            "estimates": {k: e.to_dict() for k, e in self.estimates.items()},
            "diagnostics": self.diagnostics,
        }


def run_checks(
    root: str | Path,
    campaign: str = "all",
    *,
    clip: float | None = None,
    n_boot: int = 200,
    seed: int = 0,
    cache_dir: str | Path | None = None,
    prior_path: str | Path | None = None,
) -> list[RealCheck]:
    """Every check for one campaign: BTS from random logs (hourly and weekly targets, and the published
    prior when ``prior_path`` is given), then the random policy from BTS logs."""
    rnd = read_log(root, "random", campaign, cache_dir)
    bts = read_log(root, "bts", campaign, cache_dir)
    rnd_ctr, rnd_se = clustered_mean(rnd)
    bts_ctr, bts_se = clustered_mean(bts)
    checks = []

    def run(name, log, target, truth, truth_se, logged, target_name):
        est, diag = estimate_policy(log, target, clip=clip, n_boot=n_boot, seed=seed)
        checks.append(RealCheck(name, campaign, log.policy, target_name, len(log), truth, truth_se, logged, est, diag))

    run("bts_hourly_from_random", rnd, hourly_policy(bts), bts_ctr, bts_se, rnd_ctr, "bts (hourly behaviour)")
    run("bts_weekly_from_random", rnd, empirical_policy(bts), bts_ctr, bts_se, rnd_ctr, "bts (weekly average)")
    if prior_path is not None:
        a, b = read_prior(prior_path, campaign)
        run("bts_prior_from_random", rnd, prior_policy(a, b, seed=seed), bts_ctr, bts_se, rnd_ctr, "bts (published prior)")
    run("random_from_bts", bts, uniform_policy(bts.n_items), rnd_ctr, rnd_se, bts_ctr, "random (uniform)")
    return checks
