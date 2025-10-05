"""Project-specific exception classes."""

from __future__ import annotations


class OptilbError(Exception):
    """Base class for optilb-specific exceptions."""


class EvaluationBudgetExceeded(OptilbError):
    """Raised when an optimiser exceeds the allowed number of evaluations."""

    def __init__(self, max_evals: int) -> None:
        self.max_evals = max_evals
        super().__init__(f"Evaluation budget of {max_evals} function calls exceeded")


class MissingDependencyError(OptilbError):
    """Raised when an optional dependency is required but not installed."""

    def __init__(self, dependency: str, guidance: str | None = None) -> None:
        message = (
            f"Missing optional dependency '{dependency}'."
            if guidance is None
            else f"Missing optional dependency '{dependency}': {guidance}"
        )
        self.dependency = dependency
        super().__init__(message)


__all__ = ["OptilbError", "EvaluationBudgetExceeded", "MissingDependencyError"]
