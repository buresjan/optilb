"""optilb package."""

from __future__ import annotations

from .core import Constraint, DesignPoint, DesignSpace, EvaluationRecord, OptResult
from .exceptions import (
    EvaluationBudgetExceeded,
    MissingDependencyError,
    OptilbError,
    UnknownObjectiveError,
    UnknownOptimizerError,
)
from .objectives import get_objective
from .optimizers import (
    BFGSOptimizer,
    EarlyStopper,
    MADSOptimizer,
    NelderMeadOptimizer,
    Optimizer,
)
from .problem import OptimizationLog, OptimizationProblem
from .sampling import lhs

__all__ = [
    "__version__",
    "DesignSpace",
    "DesignPoint",
    "EvaluationRecord",
    "Constraint",
    "OptResult",
    "Optimizer",
    "BFGSOptimizer",
    "MADSOptimizer",
    "NelderMeadOptimizer",
    "EarlyStopper",
    "OptilbError",
    "EvaluationBudgetExceeded",
    "MissingDependencyError",
    "lhs",
    "get_objective",
    "OptimizationProblem",
    "OptimizationLog",
    "UnknownObjectiveError",
    "UnknownOptimizerError",
]
__version__ = "0.0.0"
