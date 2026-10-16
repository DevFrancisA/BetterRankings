"""Deterministic random numbers shared with the TypeScript port.

The generator is mulberry32. Its state advances by a fixed constant on every
draw, so the k-th output is a pure function of ``seed + k * C``. That makes it
easy to vectorise with NumPy here while the browser version draws one number at
a time, and both produce the same stream bit for bit.
"""

from __future__ import annotations

import numpy as np

MASK = 0xFFFFFFFF
STEP = 0x6D2B79F5
_TWO_32 = 4294967296.0


def _hash32(a: np.ndarray) -> np.ndarray:
    """Mulberry32 output function on uint64 arrays holding uint32 values."""
    t = ((a ^ (a >> 15)) * (a | 1)) & MASK
    t = ((t + (((t ^ (t >> 7)) * (t | 61)) & MASK)) & MASK) ^ t
    return (t ^ (t >> 14)) & MASK


def fnv1a(text: str) -> int:
    """32-bit FNV-1a hash of a UTF-8 string."""
    h = 0x811C9DC5
    for byte in text.encode("utf-8"):
        h ^= byte
        h = (h * 0x01000193) & MASK
    return h


def derive_seed(seed: int, name: str) -> int:
    """Seed for a named sub-stream, so each part of a run draws independently."""
    a = np.array([(((seed & MASK) ^ fnv1a(name)) + STEP) & MASK], dtype=np.uint64)
    return int(_hash32(a)[0])


class Rng:
    """A named stream of uniforms in [0, 1) and standard normals."""

    def __init__(self, seed: int, name: str | None = None):
        self.seed = derive_seed(seed, name) if name is not None else seed & MASK
        self.count = 0

    def uniforms(self, n: int) -> np.ndarray:
        k = np.arange(self.count + 1, self.count + n + 1, dtype=np.uint64)
        a = (np.uint64(self.seed) + k * np.uint64(STEP)) & np.uint64(MASK)
        self.count += n
        return _hash32(a).astype(np.float64) / _TWO_32

    def normals(self, n: int) -> np.ndarray:
        """Box-Muller, one normal per pair of uniforms (the sine half is dropped)."""
        u = self.uniforms(2 * n).reshape(n, 2)
        return np.sqrt(-2.0 * np.log(1.0 - u[:, 0])) * np.cos(2.0 * np.pi * u[:, 1])
