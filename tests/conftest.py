import numpy as np
import pytest

from betterrankings.policies import ranking_values
from betterrankings.perms import permutations


class TinyRankingWorld:
    """Three items, no context: six rankings with exactly known click rates.

    Small enough that the true value of any policy is a dot product, and that
    hundreds of independent logs can be drawn at once.
    """

    def __init__(self, logging: np.ndarray, target: np.ndarray):
        theta = np.array([1.0, 0.5, 1.0 / 3.0])
        rel = np.array([0.6, 0.3, 0.1])
        g = (rel[:, None] * theta[None, :])[None]  # (1, slot, position)
        self.click_prob = g[0]
        self.values = ranking_values(g)[0]  # expected clicks of each of the 6 rankings
        self.perms = permutations(3)
        self.mu = np.asarray(logging, dtype=float)
        self.pi = np.asarray(target, dtype=float)
        self.truth = float(self.pi @ self.values)

    def draw(self, n_logs: int, n_rows: int, seed: int):
        """Many independent logs: chosen ranking index and total clicks, shape (n_logs, n_rows)."""
        rng = np.random.default_rng(seed)
        a = rng.choice(len(self.mu), size=(n_logs, n_rows), p=self.mu)
        p = self.click_prob[self.perms[a], np.arange(3)]  # (n_logs, n_rows, 3)
        clicks = rng.random(p.shape) < p
        return a, clicks.sum(axis=2).astype(float)


@pytest.fixture
def tiny_world():
    logging = np.array([0.30, 0.25, 0.20, 0.12, 0.08, 0.05])
    target = np.array([0.05, 0.05, 0.10, 0.10, 0.30, 0.40])
    return TinyRankingWorld(logging, target)
