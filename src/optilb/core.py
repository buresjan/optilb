from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Sequence

import numpy as np


@dataclass(slots=True, frozen=True)
class DesignSpace:
    """Continuous design space defined by lower and upper bounds."""

    lower: np.ndarray
    upper: np.ndarray
    names: Sequence[str] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "lower", np.asarray(self.lower, dtype=float))
        object.__setattr__(self, "upper", np.asarray(self.upper, dtype=float))
        if self.lower.shape != self.upper.shape:
            raise ValueError("Lower and upper bounds must have the same shape")
        if self.names is not None and len(self.names) != self.lower.size:
            raise ValueError("Number of names must match dimension")

    @property
    def dimension(self) -> int:
        return int(self.lower.size)

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, DesignSpace):
            return False
        return (
            np.array_equal(self.lower, other.lower)
            and np.array_equal(self.upper, other.upper)
            and self.names == other.names
        )


@dataclass(slots=True, frozen=True)
class DesignPoint:
    """Single design vector with optional metadata."""

    x: np.ndarray
    tag: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        object.__setattr__(self, "x", np.asarray(self.x, dtype=float))

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, DesignPoint):
            return False
        return (
            np.array_equal(self.x, other.x)
            and self.tag == other.tag
            and self.timestamp == other.timestamp
        )


@dataclass(slots=True, frozen=True)
class Constraint:
    """Feasibility or penalty function."""

    func: Callable[[np.ndarray], bool | float]
    name: str | None = None

    def __call__(self, x: np.ndarray) -> bool | float:
        return self.func(x)


@dataclass(slots=True, frozen=True)
class OptResult:
    """Result of an optimisation run."""

    best_x: np.ndarray
    best_f: float
    history: Sequence[DesignPoint] = field(default_factory=tuple)
    nfev: int = 0

    def __post_init__(self) -> None:
        arr = np.asarray(self.best_x, dtype=float)
        object.__setattr__(self, "best_x", arr)
        if not isinstance(self.history, tuple):
            object.__setattr__(self, "history", tuple(self.history))

    def __eq__(self, other: object) -> bool:  # type: ignore[override]
        if not isinstance(other, OptResult):
            return False
        return (
            np.array_equal(self.best_x, other.best_x)
            and self.best_f == other.best_f
            and self.history == other.history
            and self.nfev == other.nfev
        )
