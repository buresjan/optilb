"""optilb package."""

from __future__ import annotations

from .core import Constraint, DesignPoint, DesignSpace, OptResult

__all__ = [
    "__version__",
    "DesignSpace",
    "DesignPoint",
    "Constraint",
    "OptResult",
]
__version__ = "0.0.0"
