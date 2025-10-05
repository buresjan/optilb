from __future__ import annotations

import time
from dataclasses import dataclass, field


@dataclass
class EarlyStopper:
    """Utility to stop optimisation early based on progress."""

    eps: float = 1e-6
    patience: int = 10
    f_target: float | None = None
    time_limit: float | None = None
    enabled: bool = True

    _best_f: float = field(default=float("inf"), init=False)
    _counter: int = field(default=0, init=False)
    _start: float = field(default_factory=time.perf_counter, init=False)
    _stopped: bool = field(default=False, init=False)

    def reset(self) -> None:
        """Reset internal state for a new run."""
        self._best_f = float("inf")
        self._counter = 0
        self._start = time.perf_counter()
        self._stopped = False

    def update(self, f_val: float) -> bool:
        """Update with the current objective value.

        Returns ``True`` if stopping criteria are met.
        """
        if not self.enabled:
            return False

        if (
            self.time_limit is not None
            and (time.perf_counter() - self._start) >= self.time_limit
        ):
            self._stopped = True
            return True

        stop = False
        if self.f_target is not None and f_val <= self.f_target:
            stop = True

        if self._best_f - f_val > self.eps:
            self._best_f = f_val
            self._counter = 0
        else:
            self._counter += 1
            if self._counter >= self.patience:
                stop = True

        if stop:
            self._stopped = True
        return stop

    @property
    def stopped(self) -> bool:
        """Whether the stopper has requested an early termination."""

        return self._stopped


__all__ = ["EarlyStopper"]
