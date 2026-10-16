"""A click model learned from the log, used by the direct and doubly robust estimators.

The model is a ridge-regularised logistic regression predicting whether an
item is clicked from its features, the query context, their interactions and
the position it was shown at::

    [1, pop, v_1..v_F, x_1..x_D, x_a * v_b for all a, b, position 2..K one-hot]

It is fitted by a fixed number of Newton steps, so the result does not depend on
a stopping tolerance and the browser port reproduces it.

Cross-fitting: rows are split into folds; the model scoring a row is trained
only on the other folds, so no row is predicted by a model that saw its clicks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rng import Rng
from .world import Features, sigmoid

RIDGE = 1.0
NEWTON_STEPS = 10


def n_features(d_ctx: int, d_item: int, n_slots: int) -> int:
    return 2 + d_item + d_ctx + d_ctx * d_item + (n_slots - 1)


def design(ctx: np.ndarray, pop: np.ndarray, feats: np.ndarray, position: np.ndarray, n_slots: int) -> np.ndarray:
    """Feature rows for (context, item, 0-based position) triples, all aligned on axis 0."""
    rows = ctx.shape[0]
    d_ctx, d_item = ctx.shape[1], feats.shape[1]
    cross = (ctx[:, :, None] * feats[:, None, :]).reshape(rows, d_ctx * d_item)
    onehot = np.zeros((rows, n_slots - 1))
    has = position > 0
    onehot[np.nonzero(has)[0], position[has] - 1] = 1.0
    return np.hstack([np.ones((rows, 1)), pop[:, None], feats, ctx, cross, onehot])


def fit_logistic(x: np.ndarray, y: np.ndarray, ridge: float = RIDGE, steps: int = NEWTON_STEPS) -> np.ndarray:
    """Newton's method for ridge logistic regression; the intercept is not penalised."""
    penalty = np.full(x.shape[1], ridge)
    penalty[0] = 0.0
    beta = np.zeros(x.shape[1])
    for _ in range(steps):
        p = sigmoid(x @ beta)
        grad = x.T @ (y - p) - penalty * beta
        hess = (x * (p * (1.0 - p))[:, None]).T @ x + np.diag(penalty)
        beta = beta + np.linalg.solve(hess, grad)
    return beta


def _training_rows(f: Features, clicks: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    m = f.n_slots
    r = np.repeat(rows, m)
    k = np.tile(np.arange(m), len(rows))
    x = design(f.ctx[r], f.pop[r, k], f.feats[r, k], k, m)
    return x, clicks[r, k].astype(float)


def predict_matrix(beta: np.ndarray, f: Features) -> np.ndarray:
    """Predicted click probability of every slot at every position, shape (n, m, m)."""
    n, m = len(f), f.n_slots
    r = np.repeat(np.arange(n), m * m)
    j = np.tile(np.repeat(np.arange(m), m), n)
    k = np.tile(np.arange(m), n * m)
    x = design(f.ctx[r], f.pop[r, j], f.feats[r, j], k, m)
    return sigmoid(x @ beta).reshape(n, m, m)


def fold_ids(n: int, n_folds: int, seed: int) -> np.ndarray:
    return np.floor(Rng(seed, "folds").uniforms(n) * n_folds).astype(np.int64)


@dataclass
class CrossFit:
    q: np.ndarray  # (n, m, m) held-out predictions, slot x position
    folds: np.ndarray  # (n,)
    coefs: list[np.ndarray]

    def logged(self) -> np.ndarray:
        """Predicted clicks of the ranking that was shown (slot k at position k)."""
        m = self.q.shape[1]
        return self.q[:, np.arange(m), np.arange(m)].sum(axis=1)


def cross_fit(f: Features, clicks: np.ndarray, n_folds: int = 2, seed: int = 0) -> CrossFit:
    n = len(f)
    folds = fold_ids(n, n_folds, seed)
    q = np.zeros((n, f.n_slots, f.n_slots))
    coefs = []
    for k in range(n_folds):
        held = np.nonzero(folds == k)[0]
        train = np.nonzero(folds != k)[0]
        if len(train) == 0:  # too few rows to split; fall back to all rows
            train = np.arange(n)
        x, y = _training_rows(f, clicks, train)
        beta = fit_logistic(x, y)
        coefs.append(beta)
        if len(held):
            q[held] = predict_matrix(beta, f.take(held))
    return CrossFit(q, folds, coefs)
