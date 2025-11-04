from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from ..core import (
    Constraint,
    DesignPoint,
    DesignSpace,
    EvaluationRecord,
    OptResult,
)
from ..exceptions import EvaluationBudgetExceeded
from .early_stop import EarlyStopper


@dataclass
class _CountingFunction:
    func: Callable[[np.ndarray], float]
    optimizer: "Optimizer"
    map_input: Callable[[np.ndarray], np.ndarray] | None = None
    use_cache: bool = False
    last_val: float | None = None

    def __call__(self, x: np.ndarray) -> float:
        optimizer = self.optimizer
        arr = np.asarray(x, dtype=float)
        mapped = np.asarray(
            self.map_input(arr), dtype=float
        ) if self.map_input is not None else arr
        cache_key: bytes | None = None
        cache_event = None
        use_cache = self.use_cache and optimizer._cache_enabled
        cached_value: float | None = None
        if use_cache:
            try:
                cache_key = optimizer._make_cache_key(mapped)
                cached_value, cache_event, should_evaluate = optimizer._cache_check(
                    cache_key
                )
                if not should_evaluate and cached_value is not None:
                    with optimizer._state_lock:
                        optimizer._last_eval_point = arr.copy()
                        optimizer._update_best(arr, cached_value)
                    return cached_value
                if not should_evaluate and cached_value is None:
                    # Cache disabled due to failure; fall back to direct evaluation
                    use_cache = False
            except Exception:
                cache_key = None
                use_cache = False
        with optimizer._state_lock:
            max_evals = optimizer._max_evals
            if max_evals is not None and optimizer._nfev >= max_evals:
                optimizer._budget_exhausted = True
                raise EvaluationBudgetExceeded(max_evals)
            optimizer._nfev += 1
        try:
            val = float(self.func(x))
        except BaseException:
            with optimizer._state_lock:
                optimizer._nfev = max(0, optimizer._nfev - 1)
            if use_cache and cache_key is not None:
                optimizer._cache_fail(cache_key, cache_event)
            raise
        with optimizer._state_lock:
            self.last_val = val
            optimizer._last_eval_point = arr.copy()
            optimizer._update_best(arr, val)
            max_evals = optimizer._max_evals
            if max_evals is not None and optimizer._nfev >= max_evals:
                optimizer._budget_exhausted = True
            should_store = use_cache and cache_key is not None and np.isfinite(val)
        record_point = mapped
        optimizer._record_evaluation(record_point, val)
        if should_store and cache_key is not None:
            optimizer._cache_complete(cache_key, val, cache_event)
        elif use_cache and cache_key is not None:
            optimizer._cache_fail(cache_key, cache_event)
        return val


class Optimizer(ABC):
    """Abstract base class for local optimisers."""

    def __init__(self, *, memoize: bool = False) -> None:
        self._state_lock = threading.RLock()
        self._history: list[DesignPoint] = []
        self._evaluations: list[EvaluationRecord] = []
        self._nfev: int = 0
        self._max_evals: int | None = None
        self._budget_exhausted: bool = False
        self._last_eval_point: np.ndarray | None = None
        self._last_budget_exhausted: bool = False
        self._best_point: np.ndarray | None = None
        self._best_value: float | None = None
        self._cache_enabled: bool = bool(memoize)
        self._cache: dict[bytes, float] = {}
        self._cache_events: dict[bytes, threading.Event] = {}

    # ------------------------------------------------------------------
    # Utility methods shared by all optimisers
    # ------------------------------------------------------------------
    @property
    def history(self) -> tuple[DesignPoint, ...]:
        """Sequence of recorded design points."""
        with self._state_lock:
            return tuple(self._history)

    @property
    def evaluations(self) -> tuple[EvaluationRecord, ...]:
        """Sequence of recorded objective evaluations."""
        with self._state_lock:
            return tuple(self._evaluations)
    @property
    def nfev(self) -> int:
        """Number of objective function evaluations so far."""
        return self._nfev

    @property
    def budget_exhausted(self) -> bool:
        """Whether the evaluation budget was reached in the last run."""

        return self._budget_exhausted

    @property
    def last_budget_exhausted(self) -> bool:
        """Whether the evaluation budget was hit during the previous run."""

        return self._last_budget_exhausted

    def _configure_budget(self, max_evals: int | None) -> None:
        with self._state_lock:
            if max_evals is not None:
                if max_evals < 0:
                    raise ValueError("max_evals must be non-negative")
                self._max_evals = max_evals
            else:
                self._max_evals = None
            self._budget_exhausted = False
            self._last_budget_exhausted = False

    def _clear_budget(self) -> None:
        with self._state_lock:
            self._last_budget_exhausted = self._budget_exhausted
            self._max_evals = None
            self._budget_exhausted = False
            self._last_eval_point = None

    def _get_last_eval_point(self) -> np.ndarray | None:
        with self._state_lock:
            if self._last_eval_point is None:
                return None
            return self._last_eval_point.copy()

    def reset_history(self) -> None:
        """Clear stored optimisation history."""
        with self._state_lock:
            self._history.clear()
            self._evaluations.clear()
            self._cache.clear()
            self._cache_events.clear()
            self._nfev = 0
            self._budget_exhausted = False
            self._last_eval_point = None
            self._last_budget_exhausted = False
            self._best_point = None
            self._best_value = None

    def record(self, x: np.ndarray, tag: str | None = None) -> None:
        """Record a point in the optimisation history."""
        pt = DesignPoint(x=np.asarray(x, dtype=float), tag=tag)
        with self._state_lock:
            self._history.append(pt)

    def finalize_history(self) -> None:
        """Hook for subclasses to post-process history before consumption."""

        return None

    def _update_best(self, x: np.ndarray, value: float) -> None:
        arr = np.asarray(x, dtype=float)
        with self._state_lock:
            if self._best_value is None or value < self._best_value:
                self._best_value = float(value)
                self._best_point = arr.copy()

    def _record_evaluation(self, x: np.ndarray, value: float) -> None:
        record = EvaluationRecord(x=x, value=float(value))
        with self._state_lock:
            self._evaluations.append(record)

    def _make_cache_key(self, x: np.ndarray) -> bytes:
        if not x.flags.c_contiguous:
            x = np.ascontiguousarray(x, dtype=float)
        else:
            x = np.asarray(x, dtype=float)
        shape_bytes = np.asarray(x.shape, dtype=np.int64).tobytes()
        return shape_bytes + x.tobytes()

    def _cache_check(
        self, key: bytes, *, wait: bool = True
    ) -> tuple[float | None, threading.Event | None, bool]:
        """Return cached value/status; optionally skip waiting for in-flight work."""
        with self._state_lock:
            if key in self._cache:
                return self._cache[key], None, False
            event = self._cache_events.get(key)
            if event is None:
                event = threading.Event()
                self._cache_events[key] = event
                return None, event, True
        if wait:
            event.wait()
            with self._state_lock:
                return self._cache.get(key), None, False
        return None, event, False

    def _cache_complete(
        self, key: bytes, value: float, event: threading.Event | None
    ) -> None:
        with self._state_lock:
            self._cache[key] = float(value)
            if event is not None:
                self._cache_events.pop(key, None)
        if event is not None:
            event.set()

    def _cache_fail(self, key: bytes, event: threading.Event | None) -> None:
        if event is None:
            return
        with self._state_lock:
            current = self._cache_events.get(key)
            if current is event:
                self._cache_events.pop(key, None)
        event.set()

    def _cache_value(self, key: bytes) -> float | None:
        with self._state_lock:
            return self._cache.get(key)

    def _get_best_evaluation(self) -> tuple[np.ndarray | None, float | None]:
        with self._state_lock:
            if self._best_point is None or self._best_value is None:
                return None, None
            return self._best_point.copy(), float(self._best_value)

    def _validate_x0(self, x0: np.ndarray, space: DesignSpace) -> np.ndarray:
        """Ensure ``x0`` matches the design space."""
        arr = np.asarray(x0, dtype=float)
        if arr.shape != space.lower.shape:
            raise ValueError("Initial point has wrong dimension")
        if np.any(arr < space.lower) or np.any(arr > space.upper):
            raise ValueError("Initial point outside design bounds")
        return arr

    def _wrap_objective(
        self,
        objective: Callable[[np.ndarray], float],
        *,
        map_input: Callable[[np.ndarray], np.ndarray] | None = None,
        use_cache: bool | None = None,
    ) -> _CountingFunction:
        """Return objective wrapper that increments the evaluation counter."""

        return _CountingFunction(
            func=objective,
            optimizer=self,
            map_input=map_input,
            use_cache=self._cache_enabled if use_cache is None else bool(use_cache),
        )

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
        max_evals: int | None = None,
        tol: float = 1e-6,
        seed: int | None = None,
        parallel: bool = False,
        verbose: bool = False,
        early_stopper: EarlyStopper | None = None,
    ) -> OptResult:
        """Run optimisation and return the result."""
