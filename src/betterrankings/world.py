"""The simulated ranking world: the ground truth estimators are checked against.

Each query has a context vector ``x`` and a short list of candidate items drawn
from a catalog. Every item has a popularity score and a feature vector ``v``.
The true chance that a user clicks item ``j`` shown at position ``k`` follows a
position-based model::

    P(click) = theta_k * relevance(x, j)
    theta_k  = 1 / k ** position_decay            (attention falls with rank)
    relevance(x, j) = sigmoid(pop_weight * pop_j + x' W v_j + base_logit)

Because the simulator knows these probabilities exactly, it can compute the
expected clicks per query of any policy without sampling clicks. That is the
"true online test" each estimate is compared with.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rng import Rng


@dataclass(frozen=True)
class WorldConfig:
    n_items: int = 200
    n_slots: int = 4
    d_ctx: int = 3
    d_item: int = 3
    position_decay: float = 1.0
    pop_weight: float = 0.6
    base_logit: float = -1.0
    # The hidden preference matrix and the model rankers' weights are fixed
    # across runs; a run seed changes the catalog, the traffic and the clicks.
    world_seed: int = 1729

    def position_weights(self) -> np.ndarray:
        return 1.0 / np.arange(1, self.n_slots + 1, dtype=np.float64) ** self.position_decay


@dataclass(frozen=True)
class WorldParams:
    w_true: np.ndarray  # (d_ctx, d_item) hidden preferences
    w_model: np.ndarray  # a good but imperfect estimate of w_true
    w_log: np.ndarray  # the weaker model behind the logging policy


@dataclass(frozen=True)
class Catalog:
    pop: np.ndarray  # (n_items,)
    feats: np.ndarray  # (n_items, d_item)


@dataclass
class Features:
    """Per-query inputs, with candidates in slot order."""

    ctx: np.ndarray  # (n, d_ctx)
    pop: np.ndarray  # (n, m)
    feats: np.ndarray  # (n, m, d_item)

    def __len__(self) -> int:
        return self.ctx.shape[0]

    @property
    def n_slots(self) -> int:
        return self.pop.shape[1]

    def take(self, rows: np.ndarray) -> "Features":
        return Features(self.ctx[rows], self.pop[rows], self.feats[rows])


def make_params(cfg: WorldConfig) -> WorldParams:
    r = Rng(cfg.world_seed, "params")
    d = cfg.d_ctx * cfg.d_item
    shape = (cfg.d_ctx, cfg.d_item)
    w_true = r.normals(d).reshape(shape)
    w_model = w_true + 0.35 * r.normals(d).reshape(shape)
    w_log = 0.5 * w_true + 0.5 * r.normals(d).reshape(shape)
    return WorldParams(w_true, w_model, w_log)


def make_catalog(cfg: WorldConfig, seed: int) -> Catalog:
    r = Rng(seed, "catalog")
    pop = r.normals(cfg.n_items)
    feats = r.normals(cfg.n_items * cfg.d_item).reshape(cfg.n_items, cfg.d_item)
    return Catalog(pop, feats)


def sample_queries(
    cfg: WorldConfig, catalog: Catalog, n: int, seed: int, prefix: str
) -> tuple[Features, np.ndarray]:
    """Draw ``n`` queries: contexts plus distinct candidate items for each.

    Candidates come from a partial Fisher-Yates shuffle of the catalog, one
    uniform per slot. Returns the features and the candidate item ids (n, m).
    """
    m, big_n = cfg.n_slots, cfg.n_items
    if m > big_n:
        raise ValueError("the catalog must hold at least as many items as list slots")
    ctx = Rng(seed, f"{prefix}-contexts").normals(n * cfg.d_ctx).reshape(n, cfg.d_ctx)
    u = Rng(seed, f"{prefix}-candidates").uniforms(n * m).reshape(n, m)
    deck = np.tile(np.arange(big_n, dtype=np.int64), (n, 1))
    rows = np.arange(n)
    for k in range(m):
        j = k + np.floor(u[:, k] * (big_n - k)).astype(np.int64)
        held = deck[rows, k].copy()
        deck[rows, k] = deck[rows, j]
        deck[rows, j] = held
    ids = deck[:, :m]
    return Features(ctx, catalog.pop[ids], catalog.feats[ids]), ids


def bilinear(ctx: np.ndarray, w: np.ndarray, feats: np.ndarray) -> np.ndarray:
    """x' W v for every query and candidate, shape (n, m)."""
    xw = ctx @ w  # (n, d_item)
    return (feats * xw[:, None, :]).sum(axis=2)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def true_relevance(cfg: WorldConfig, params: WorldParams, f: Features) -> np.ndarray:
    return sigmoid(cfg.pop_weight * f.pop + bilinear(f.ctx, params.w_true, f.feats) + cfg.base_logit)


def true_click_matrix(cfg: WorldConfig, params: WorldParams, f: Features) -> np.ndarray:
    """Exact click probability of slot j shown at position k, shape (n, m, m)."""
    rel = true_relevance(cfg, params, f)
    theta = 1.0 / np.arange(1, f.n_slots + 1, dtype=np.float64) ** cfg.position_decay
    return rel[:, :, None] * theta[None, None, :]
