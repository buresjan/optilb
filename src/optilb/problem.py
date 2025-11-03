from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Sequence

import numpy as np

from .core import Constraint, DesignSpace, OptResult
from .exceptions import UnknownOptimizerError
from .optimizers import (
    BFGSOptimizer,
    EarlyStopper,
    MADSOptimizer,
    NelderMeadOptimizer,
    Optimizer,
)

logger = logging.getLogger("optilb")


@dataclass(slots=True)
class OptimizationLog:
    """Summary information for a completed optimisation run."""

    optimizer: str
    runtime: float
    nfev: int
    early_stopped: bool = False


@dataclass(slots=True)
class _EvalCap:
    """Objective wrapper enforcing a maximum number of evaluations."""

    func: Callable[[np.ndarray], float]
    limit: int
    calls: int = 0
    best_f: float = field(default=float("inf"), init=False)
    best_x: np.ndarray | None = field(default=None, init=False)

    def __call__(self, x: np.ndarray) -> float:
        if self.calls >= self.limit:
            raise StopIteration
        val = float(self.func(x))
        self.calls += 1
        if val < self.best_f:
            self.best_f = val
            self.best_x = np.asarray(x, dtype=float).copy()
        return val


class OptimizationProblem:
    """Unified façade to run different local optimisers.

    Args:
        objective: Objective function returning a scalar value.
        space: Continuous design space defining variable bounds.
        x0: Starting point for the search.
        optimizer: Optimiser name or instance. Supported names are
            ``"bfgs"``, ``"nelder-mead"`` and ``"mads"``. If an instance is
            provided it is used directly.
        constraints: Optional sequence of constraints.
        parallel: Evaluate independent points concurrently where supported.
        normalize: Whether to operate in the unit hypercube when supported.
            ``None`` keeps the optimiser's default.
        max_iter: Maximum number of iterations.
        tol: Convergence tolerance.
        seed: Random seed for reproducibility.
        max_evals: Hard cap on objective evaluations. When reached, the best
            known point is returned and ``early_stopped`` is set in the log.
        early_stopper: Optional early stopping controller.
        verbose: Emit progress information.
        optimizer_options: Keyword arguments used to construct the optimiser
            when ``optimizer`` is a string identifier.
        optimize_options: Extra keyword arguments forwarded to
            ``optimizer.optimize``.

    Returns:
        None

    Raises:
        ValueError: If an unknown optimizer name is provided.

    Examples:
        >>> from optilb.core import DesignSpace
        >>> def sphere(x):
        ...     return float((x ** 2).sum())
        >>> space = DesignSpace(lower=[-1, -1], upper=[1, 1])
        >>> problem = OptimizationProblem(objective=sphere, space=space, x0=[0.5, 0.5])
        >>> result = problem.run()
        >>> result.best_x.round(1)
        array([0., 0.])
    """

    def __init__(
        self,
        objective: Callable[[np.ndarray], float],
        space: DesignSpace,
        x0: np.ndarray | Sequence[float],
        *,
        optimizer: str | Optimizer = "bfgs",
        constraints: Sequence[Constraint] | None = None,
        parallel: bool = False,
        normalize: bool | None = None,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int | None = None,
        max_evals: int | None = None,
        early_stopper: EarlyStopper | None = None,
        verbose: bool = False,
        optimizer_options: dict[str, Any] | None = None,
        optimize_options: dict[str, Any] | None = None,
    ) -> None:
        self.objective = objective
        self.space = space
        self.x0 = np.asarray(x0, dtype=float)
        self.constraints = tuple(constraints or ())
        self.parallel = parallel
        self.normalize = normalize
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.max_evals = max_evals
        self.early_stopper = early_stopper
        self.verbose = verbose
        self.optimize_options = dict(optimize_options or {})

        opt_opts = dict(optimizer_options or {})
        self.optimizer: Optimizer
        if isinstance(optimizer, str):
            key = optimizer.lower()
            if key in {"bfgs", "lbfgsb", "l-bfgs-b"}:
                self.optimizer = BFGSOptimizer(**opt_opts)
            elif key in {"nelder-mead", "nm"}:
                self.optimizer = NelderMeadOptimizer(**opt_opts)
            elif key == "mads":
                self.optimizer = MADSOptimizer(**opt_opts)
            else:  # pragma: no cover - defensive
                raise UnknownOptimizerError(f"Unknown optimizer '{optimizer}'")
        else:
            self.optimizer = optimizer
            if opt_opts:
                logger.warning(
                    "optimizer_options ignored when optimizer is an instance"
                )

        self.log: OptimizationLog | None = None
        self._result: OptResult | None = None

    # ------------------------------------------------------------------
    def _build_optimize_kwargs(self) -> dict[str, Any]:
        sig = inspect.signature(self.optimizer.optimize)
        kwargs: dict[str, Any] = {}
        for name, value in {
            "max_iter": self.max_iter,
            "max_evals": self.max_evals,
            "tol": self.tol,
            "seed": self.seed,
            "parallel": self.parallel,
            "verbose": self.verbose,
            "early_stopper": self.early_stopper,
        }.items():
            if name in sig.parameters:
                kwargs[name] = value
        if self.normalize is not None and "normalize" in sig.parameters:
            kwargs["normalize"] = self.normalize
        kwargs.update(self.optimize_options)
        return kwargs

    # ------------------------------------------------------------------
    def run(self) -> OptResult:
        """Execute the optimisation and return the result."""

        if self.max_evals == 0:
            x0 = self.optimizer._validate_x0(self.x0, self.space)
            self.optimizer.reset_history()
            self.optimizer.record(x0, tag="start")
            self.optimizer.record(x0, tag="cap")
            res = OptResult(
                best_x=x0,
                best_f=float("inf"),
                history=self.optimizer.history,
                evaluations=self.optimizer.evaluations,
                nfev=0,
            )
            self.log = OptimizationLog(
                optimizer=type(self.optimizer).__name__,
                runtime=0.0,
                nfev=0,
                early_stopped=True,
            )
            self._result = res
            return res

        kwargs = self._build_optimize_kwargs()
        objective: Callable[[np.ndarray], float] = self.objective
        capper: _EvalCap | None = None
        if self.max_evals is not None:
            capper = _EvalCap(objective, self.max_evals)
            objective = capper

        start = perf_counter()
        early = False
        try:
            res = self.optimizer.optimize(
                objective=objective,
                x0=self.x0,
                space=self.space,
                constraints=self.constraints,
                **kwargs,
            )
        except StopIteration:
            early = True
            assert capper is not None
            best_x = capper.best_x
            best_f = capper.best_f
            if best_x is None:
                best_x = np.asarray(self.x0, dtype=float)
            best_f = float(best_f)
            self.optimizer.finalize_history()
            self.optimizer.record(best_x, tag="cap")
            res = OptResult(
                best_x=best_x,
                best_f=best_f,
                history=self.optimizer.history,
                evaluations=self.optimizer.evaluations,
                nfev=self.optimizer.nfev,
            )
        runtime = perf_counter() - start
        if self.early_stopper is not None and self.early_stopper.stopped:
            early = True
        if getattr(self.optimizer, "last_budget_exhausted", False):
            early = True
        self.log = OptimizationLog(
            optimizer=type(self.optimizer).__name__,
            runtime=runtime,
            nfev=res.nfev,
            early_stopped=early,
        )
        self._result = res
        return res


__all__ = ["OptimizationProblem", "OptimizationLog"]
