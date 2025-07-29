"""optilb package."""

from __future__ import annotations

from .core import Constraint, DesignPoint, DesignSpace, OptResult
from .objectives import get_objective
from .sampling import lhs

__all__ = [
    "__version__",
    "DesignSpace",
    "DesignPoint",
    "Constraint",
    "OptResult",
    "lhs",
    "get_objective",
]
__version__ = "0.0.0"
