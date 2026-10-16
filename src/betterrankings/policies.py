"""Ranking policies.

Every policy maps a query's candidates to a probability for each possible
ranking, so it can report how likely it was to show any logged ranking. Three
families cover all built-in policies:

* greedy: sort by a score, probability 1 for that ranking and 0 for the rest;
* Plackett-Luce: pick the top item with probability proportional to
  exp(score / temperature), then the next from the rest, and so on;
* exploring mixture: Plackett-Luce with probability 1 - epsilon, otherwise a
  ranking chosen uniformly at random. The uniform part guarantees every
  ranking has some chance of being logged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .perms import perm_index, permutations
from .world import Features, WorldConfig, WorldParams, bilinear

LOGGING_TEMPERATURE = 1.0
LOGGING_POP_WEIGHT = 0.8
SOFT_TEMPERATURE = 0.5
NOISE_EXTRA_EXPLORATION = 0.25
NOISE_TEMPERATURE_FACTOR = 2.0
_CHUNK = 4096


@dataclass(frozen=True)
class PolicyInfo:
    id: str
    name: str
    description: str


POLICIES: tuple[PolicyInfo, ...] = (
    PolicyInfo(
        "current",
        "Current policy (logging)",
        "The policy that collected the log. A sanity check: every estimator should recover it.",
    ),
    PolicyInfo("popularity", "Popularity ranker", "Sorts candidates by global popularity, ignoring the query."),
    PolicyInfo(
        "relevance",
        "Relevance model ranker",
        "Sorts by a learned relevance model that is close to, but not exactly, the truth.",
    ),
    PolicyInfo(
        "relevance_soft",
        "Soft relevance ranker",
        "Samples rankings from the relevance model with some randomness.",
    ),
    PolicyInfo(
        "noisy_logging",
        "Noisier logging policy",
        "The logging policy with more exploration and a flatter score distribution.",
    ),
    PolicyInfo(
        "reversed",
        "Reversed relevance ranker",
        "Puts the least relevant items first. Deliberately worse.",
    ),
)
POLICY_IDS = tuple(p.id for p in POLICIES)
POLICY_NAMES = {p.id: p.name for p in POLICIES}


def logging_scores(f: Features, params: WorldParams) -> np.ndarray:
    return LOGGING_POP_WEIGHT * f.pop + bilinear(f.ctx, params.w_log, f.feats)


def relevance_scores(f: Features, cfg: WorldConfig, params: WorldParams) -> np.ndarray:
    return cfg.pop_weight * f.pop + bilinear(f.ctx, params.w_model, f.feats)


def plackett_luce_probs(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Probability of every ranking under Plackett-Luce, shape (n, m!)."""
    n, m = scores.shape
    perms = permutations(m)
    out = np.empty((n, perms.shape[0]))
    for start in range(0, n, _CHUNK):
        s = scores[start : start + _CHUNK] / temperature
        sp = s[:, perms]  # (c, P, m), score of the item at each position
        logp = np.zeros(sp.shape[:2])
        for k in range(m - 1):
            tail = sp[:, :, k:]
            mx = tail.max(axis=2)
            lse = mx + np.log(np.exp(tail - mx[:, :, None]).sum(axis=2))
            logp += sp[:, :, k] - lse
        out[start : start + _CHUNK] = np.exp(logp)
    return out


def greedy_probs(scores: np.ndarray) -> np.ndarray:
    """All mass on the ranking that sorts scores high to low (ties: lower slot first)."""
    n, m = scores.shape
    order = np.argsort(-scores, axis=1, kind="stable")
    out = np.zeros((n, permutations(m).shape[0]))
    out[np.arange(n), perm_index(order)] = 1.0
    return out


def mixture_probs(scores: np.ndarray, temperature: float, epsilon: float) -> np.ndarray:
    pl = plackett_luce_probs(scores, temperature)
    return (1.0 - epsilon) * pl + epsilon / pl.shape[1]


def noisy_exploration(exploration: float) -> float:
    return min(1.0, exploration + NOISE_EXTRA_EXPLORATION)


def policy_probs(
    policy_id: str, f: Features, cfg: WorldConfig, params: WorldParams, exploration: float
) -> np.ndarray:
    """Probability each policy assigns to every ranking of each query's candidates."""
    if policy_id == "current":
        return mixture_probs(logging_scores(f, params), LOGGING_TEMPERATURE, exploration)
    if policy_id == "noisy_logging":
        return mixture_probs(
            logging_scores(f, params),
            LOGGING_TEMPERATURE * NOISE_TEMPERATURE_FACTOR,
            noisy_exploration(exploration),
        )
    if policy_id == "popularity":
        return greedy_probs(f.pop)
    if policy_id == "relevance":
        return greedy_probs(relevance_scores(f, cfg, params))
    if policy_id == "relevance_soft":
        return plackett_luce_probs(relevance_scores(f, cfg, params), SOFT_TEMPERATURE)
    if policy_id == "reversed":
        return greedy_probs(-relevance_scores(f, cfg, params))
    raise KeyError(f"unknown policy {policy_id!r}; choose from {', '.join(POLICY_IDS)}")


def ranking_values(click_matrix: np.ndarray) -> np.ndarray:
    """Expected clicks of every ranking given a (n, m, m) slot-by-position matrix."""
    m = click_matrix.shape[1]
    perms = permutations(m)
    vals = np.zeros((click_matrix.shape[0], perms.shape[0]))
    for k in range(m):
        vals += click_matrix[:, perms[:, k], k]
    return vals


def expected_clicks(probs: np.ndarray, click_matrix: np.ndarray) -> np.ndarray:
    """Expected clicks per query when rankings are drawn from ``probs``, shape (n,)."""
    return (probs * ranking_values(click_matrix)).sum(axis=1)
