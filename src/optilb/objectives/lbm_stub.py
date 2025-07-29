from __future__ import annotations

import time
from typing import Sequence

import numpy as np


def lbm_stub(
    x: np.ndarray,
    *,
    sleep_ms: int = 0,
    centers: Sequence[float] | None = None,
    width: float = 0.2,
) -> float:
    """Lightweight surrogate for CFD cost.

    This deterministic function mimics a multi-modal CFD response using
    a sum of shifted Gaussians with additional sinusoidal perturbation.

    Parameters
    ----------
    x:
        Input vector.
    sleep_ms:
        Optional artificial delay in milliseconds to emulate wall time.
    centers:
        Locations of Gaussian peaks for each dimension.  If ``None``,
        centers are placed evenly in ``[-0.5, 0.5]``.
    width:
        Standard deviation of each Gaussian peak.

    Returns
    -------
    float
        Pseudo CFD objective value.
    """
    arr = np.asarray(x, dtype=float)
    dim = arr.size
    if centers is None:
        centers = np.linspace(-0.5, 0.5, dim)
    centers_arr = np.asarray(list(centers), dtype=float)
    if centers_arr.size != dim:
        raise ValueError("Length of centers must match dimension")

    gaussians = np.exp(-((arr - centers_arr) ** 2) / (2 * width**2))
    value = float(np.sum(gaussians))
    value += float(0.1 * np.sin(5 * np.sum(arr)))

    if sleep_ms > 0:
        time.sleep(sleep_ms / 1000.0)
    return value
