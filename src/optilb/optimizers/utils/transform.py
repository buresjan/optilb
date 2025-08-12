from __future__ import annotations

import numpy as np

from ...core import DesignSpace


class SpaceTransform:
    """Affine transform between original coordinates and [0,1]^d."""

    def __init__(self, space: DesignSpace) -> None:
        lower = np.asarray(space.lower, dtype=float)
        upper = np.asarray(space.upper, dtype=float)
        if lower.shape != upper.shape:
            raise ValueError("DesignSpace lower/upper must have the same shape")
        span = upper - lower
        if not (np.all(np.isfinite(lower)) and np.all(np.isfinite(upper))):
            raise ValueError("Normalization requires finite bounds")
        if np.any(span <= 0.0):
            raise ValueError(
                "Normalization requires strictly positive span per dimension"
            )
        self.lower = lower
        self.upper = upper
        self.span = span

    def to_unit(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if x.shape != self.lower.shape:
            raise ValueError("x has wrong dimension for SpaceTransform")
        return (x - self.lower) / self.span

    def from_unit(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=float)
        if u.shape != self.lower.shape:
            raise ValueError("u has wrong dimension for SpaceTransform")
        return self.lower + u * self.span
