"""Enumerating rankings.

A ranking of ``m`` candidates is a permutation. Lists here are short (2 to 6
items), so every possible ranking can be listed and scored exactly. Rankings are
indexed in lexicographic order, the order ``itertools.permutations`` yields.
"""

from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np

MAX_SLOTS = 6


@lru_cache(maxsize=None)
def permutations(m: int) -> np.ndarray:
    """All rankings of ``m`` slots, shape (m!, m); row p lists slot per position."""
    if not 1 <= m <= MAX_SLOTS:
        raise ValueError(f"lists must have between 1 and {MAX_SLOTS} items, got {m}")
    out = np.array(list(itertools.permutations(range(m))), dtype=np.int64)
    out.setflags(write=False)
    return out


def perm_index(order: np.ndarray) -> np.ndarray:
    """Lexicographic index (Lehmer code) of each row of ``order``, shape (n,)."""
    order = np.atleast_2d(order)
    n, m = order.shape
    idx = np.zeros(n, dtype=np.int64)
    for k in range(m):
        smaller_later = (order[:, k + 1 :] < order[:, k : k + 1]).sum(axis=1)
        idx += smaller_later * math.factorial(m - 1 - k)
    return idx
