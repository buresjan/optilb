from __future__ import annotations


class UnknownObjectiveError(KeyError):
    """Raised when an objective name is not recognised."""

    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.name = name

    def __str__(self) -> str:  # pragma: no cover - simple formatting
        return f"Unknown objective '{self.name}'"


__all__ = ["UnknownObjectiveError"]
