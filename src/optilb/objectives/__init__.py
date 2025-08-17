from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from ..exceptions import UnknownObjectiveError
from .lbm_stub import lbm_stub


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


@dataclass(slots=True)
class _NoisyDiscontinuous:
    sigma: float = 0.1
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=float)
        val = float(np.sum(np.floor(arr)))
        noise = float(self.rng.normal(0.0, self.sigma)) if self.sigma > 0 else 0.0
        return val + noise


def make_noisy_discontinuous(
    sigma: float = 0.1, *, seed: int | None = None
) -> Callable[[np.ndarray], float]:
    """Create a noisy discontinuous objective."""

    return _NoisyDiscontinuous(sigma=sigma, seed=seed)


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


@dataclass(slots=True)
class _SpikySine:
    sigma: float = 0.1
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, x: np.ndarray) -> float:
        t = float(np.asarray(x, dtype=float)[0])
        base = np.sin(5 * t) + 0.5 * np.sign(np.sin(20 * t))
        noise = float(self.rng.normal(0.0, self.sigma)) if self.sigma > 0 else 0.0
        return float(base + noise)


def make_spiky_sine(
    sigma: float = 0.1, *, seed: int | None = None
) -> Callable[[np.ndarray], float]:
    """Noisy sine with discontinuous spikes."""

    return _SpikySine(sigma=sigma, seed=seed)


@dataclass(slots=True)
class _Checkerboard:
    sigma: float = 0.05
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=float)
        if arr.size < 2:
            raise ValueError("checkerboard requires at least 2 dimensions")
        val = np.sign(np.sin(3 * arr[0])) * np.sign(np.sin(3 * arr[1]))
        noise = float(self.rng.normal(0.0, self.sigma)) if self.sigma > 0 else 0.0
        return float(val + noise)


def make_checkerboard(
    sigma: float = 0.05, *, seed: int | None = None
) -> Callable[[np.ndarray], float]:
    """Piecewise checkerboard pattern with noise."""

    return _Checkerboard(sigma=sigma, seed=seed)


@dataclass(slots=True)
class _StepRastrigin:
    sigma: float = 0.05
    seed: int | None = None
    rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def __call__(self, x: np.ndarray) -> float:
        arr = np.asarray(x, dtype=float)
        val = rastrigin(np.floor(arr))
        noise = float(self.rng.normal(0.0, self.sigma)) if self.sigma > 0 else 0.0
        return float(val + noise)


def make_step_rastrigin(
    sigma: float = 0.05, *, seed: int | None = None
) -> Callable[[np.ndarray], float]:
    """Rastrigin evaluated on floored inputs with noise."""

    return _StepRastrigin(sigma=sigma, seed=seed)


_OBJECTIVES: dict[str, Callable[..., Callable[[np.ndarray], float]]] = {
    "quadratic": lambda **_: quadratic_bowl,
    "rastrigin": lambda **_: rastrigin,
    "noisy_discontinuous": make_noisy_discontinuous,
    "spiky_sine": make_spiky_sine,
    "checkerboard": make_checkerboard,
    "step_rastrigin": make_step_rastrigin,
    "noisy_step_rastrigin": make_step_rastrigin,
    "plateau_cliff": lambda **_: plateau_cliff,
    "lbm": lambda **_: lbm_stub,
    "lbm_stub": lambda **_: lbm_stub,
}


def get_objective(name: str, **kwargs: Any) -> Callable[[np.ndarray], float]:
    """Return an objective function by name.

    Args:
        name: Identifier of the objective.
        **kwargs: Parameters passed to the objective factory.

    Returns:
        Callable[[np.ndarray], float]: Objective function.
    """
    key = name.lower()
    try:
        factory = _OBJECTIVES[key]
    except KeyError as err:
        raise UnknownObjectiveError(f"Unknown objective '{name}'") from err
    return factory(**kwargs)


__all__ = [
    "quadratic_bowl",
    "rastrigin",
    "make_noisy_discontinuous",
    "make_spiky_sine",
    "make_checkerboard",
    "make_step_rastrigin",
    "plateau_cliff",
    "lbm_stub",
    "get_objective",
]
