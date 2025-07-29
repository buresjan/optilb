from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence

import numpy as np

from ..core import Constraint, DesignPoint, DesignSpace, OptResult
from .early_stop import EarlyStopper


class Optimizer(ABC):
    """Abstract base class for local optimisers."""

    def __init__(self) -> None:
        self._history: list[DesignPoint] = []

    # ------------------------------------------------------------------
    # Utility methods shared by all optimisers
    # ------------------------------------------------------------------
    @property
    def history(self) -> tuple[DesignPoint, ...]:
        """Sequence of recorded design points."""
        return tuple(self._history)

    def reset_history(self) -> None:
        """Clear stored optimisation history."""
        self._history.clear()

    def record(self, x: np.ndarray, tag: str | None = None) -> None:
        """Record a point in the optimisation history."""
        self._history.append(DesignPoint(x=np.asarray(x, dtype=float), tag=tag))

    def _validate_x0(self, x0: np.ndarray, space: DesignSpace) -> np.ndarray:
        """Ensure ``x0`` matches the design space."""
        arr = np.asarray(x0, dtype=float)
        if arr.shape != space.lower.shape:
            raise ValueError("Initial point has wrong dimension")
        if np.any(arr < space.lower) or np.any(arr > space.upper):
            raise ValueError("Initial point outside design bounds")
        return arr

    # ------------------------------------------------------------------
    # Main optimisation hook
    # ------------------------------------------------------------------
    @abstractmethod
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        space: DesignSpace,
        constraints: Sequence[Constraint] = (),
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int | None = None,
        parallel: bool = False,
        verbose: bool = False,
        early_stopper: EarlyStopper | None = None,
    ) -> OptResult:
        """Run optimisation and return the result."""
