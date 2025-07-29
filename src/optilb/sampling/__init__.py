from __future__ import annotations

import numpy as np
from scipy.stats import qmc

from ..core import DesignPoint, DesignSpace


def lhs(
    sample_count: int,
    design_space: DesignSpace,
    seed: int | None = None,
    *,
    centered: bool = False,
    scramble: bool = True,
) -> list[DesignPoint]:
    """Generate Latin-Hypercube samples.

    Parameters
    ----------
    sample_count:
        Number of design points to generate.
    design_space:
        Continuous design space describing bounds.
    seed:
        Random seed for reproducibility.
    centered:
        Place points at the center of each hypercube cell.
    scramble:
        Enable scrambling of the unit hypercube.

    Returns
    -------
    list[DesignPoint]
        Generated design points scaled to ``design_space``.
    """
    rng = np.random.default_rng(seed)

    if centered:
        scramble = False

    sampler = qmc.LatinHypercube(d=design_space.dimension, scramble=scramble, seed=rng)
    sample = sampler.random(n=sample_count)
    scaled = qmc.scale(sample, design_space.lower, design_space.upper)

    # Round integers if bounds are integers
    rounded = scaled.copy()
    for i, (lo, hi) in enumerate(zip(design_space.lower, design_space.upper)):
        if float(lo).is_integer() and float(hi).is_integer():
            rounded[:, i] = np.rint(rounded[:, i])

    points = [DesignPoint(x=row) for row in rounded]
    return points
