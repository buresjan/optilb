from __future__ import annotations

"""Custom exception types used throughout the optilb package."""


class UnknownObjectiveError(ValueError):
    """Raised when an objective identifier is not recognised."""


class UnknownOptimizerError(ValueError):
    """Raised when an optimizer identifier is not recognised."""
