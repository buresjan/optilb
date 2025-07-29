from __future__ import annotations

from typing import Callable

import numpy as np


def quadratic_bowl(x: np.ndarray) -> float:
    """Quadratic bowl function.

    Computes ``f(x) = sum(x_i^2)``.

    The global minimum is ``0`` at ``x = 0``.

    Example:
        >>> quadratic_bowl(np.zeros(2))
        0.0
    """
    x = np.asarray(x, dtype=float)
    return float(np.sum(x**2))


def rastrigin(x: np.ndarray) -> float:
    """Rastrigin benchmark function.

    Computes ``f(x) = 10 * n + sum(x_i^2 - 10*cos(2*pi*x_i))``.

    The global minimum is ``0`` at ``x = 0``.

    Example:
        >>> rastrigin(np.zeros(3))
        0.0
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    return float(10 * n + np.sum(x**2 - 10 * np.cos(2 * np.pi * x)))


def make_noisy_discontinuous(
    sigma: float = 0.1, *, seed: int | None = None
) -> Callable[[np.ndarray], float]:
    """Create a noisy discontinuous objective.

    The base function is ``f(x) = sum(floor(x_i))``. Gaussian noise is added
    afterwards. The global minimum is ``0`` for ``x`` in ``[0, 1)^n``.

    Args:
        sigma: Standard deviation of the added Gaussian noise.
        seed: Optional random seed for reproducibility.

    Example:
        >>> f = make_noisy_discontinuous(sigma=0.0)
        >>> f(np.array([0.5, 0.5]))
        0.0
    """

    rng = np.random.default_rng(seed)

    def _func(x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=float)
        val = float(np.sum(np.floor(arr)))
        noise = float(rng.normal(0.0, sigma)) if sigma > 0 else 0.0
        return val + noise

    return _func


def plateau_cliff(x: np.ndarray) -> float:
    """Piecewise plateau with a sharp cliff.

    For ``t = x[0]``::

        f(t) = 1.0                 if t <= 0
             = 1.0 - t             if 0 < t < 1
             = 0.0                 if t >= 1

    The global minimum is ``0`` for ``t >= 1``.

    Example:
        >>> plateau_cliff(np.array([1.5]))
        0.0
    """
    t = float(np.asarray(x, dtype=float)[0])
    if t <= 0.0:
        return 1.0
    if t < 1.0:
        return 1.0 - t
    return 0.0


Objective = Callable[[np.ndarray], float]


def get_objective(name: str, **kwargs) -> Callable[[np.ndarray], float]:
    """Return an objective function by name.

    Args:
        name: Identifier of the objective.
        **kwargs: Parameters passed to the objective factory.

    Returns:
        Callable[[np.ndarray], float]: Objective function.
    """
    key = name.lower()
    if key == "quadratic":
        return quadratic_bowl
    if key == "rastrigin":
        return rastrigin
    if key == "noisy_discontinuous":
        return make_noisy_discontinuous(**kwargs)
    if key == "plateau_cliff":
        return plateau_cliff
    raise ValueError(f"Unknown objective '{name}'")


__all__ = [
    "quadratic_bowl",
    "rastrigin",
    "make_noisy_discontinuous",
    "plateau_cliff",
    "get_objective",
]
