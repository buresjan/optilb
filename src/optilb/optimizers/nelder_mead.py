from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Callable, Iterable, Sequence

import numpy as np

from ..core import Constraint, DesignSpace, OptResult
from .base import Optimizer


def _evaluate_point(
    x: np.ndarray,
    objective: Callable[[np.ndarray], float],
    lower: np.ndarray,
    upper: np.ndarray,
    constraints: Sequence[Constraint],
    penalty: float,
) -> float:
    if np.any(x < lower) or np.any(x > upper):
        return penalty
    for c in constraints:
        val = c(x)
        if isinstance(val, bool):
            if not val:
                return penalty
        else:
            if float(val) > 0.0:
                return penalty
    return float(objective(x))


logger = logging.getLogger("optilb")


class NelderMeadOptimizer(Optimizer):
    """Parallel Nelder–Mead optimiser."""

    def __init__(
        self,
        *,
        step: float | Sequence[float] = 0.5,
        alpha: float = 1.0,
        gamma: float = 2.0,
        beta: float = 0.5,
        delta: float = 0.5,
        sigma: float = 0.5,
        no_improve_thr: float = 1e-6,
        no_improv_break: int = 10,
        penalty: float = 1e12,
        n_workers: int | None = None,
    ) -> None:
        super().__init__()
        self.step = step
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.delta = delta
        self.sigma = sigma
        self.no_improve_thr = no_improve_thr
        self.no_improv_break = no_improv_break
        self.penalty = penalty
        self.n_workers = n_workers

    # ------------------------------------------------------------------
    def _make_penalised(
        self,
        objective: Callable[[np.ndarray], float],
        space: DesignSpace,
        constraints: Sequence[Constraint],
    ) -> Callable[[np.ndarray], float]:
        return partial(
            _evaluate_point,
            objective=objective,
            lower=space.lower,
            upper=space.upper,
            constraints=constraints,
            penalty=self.penalty,
        )

    def _eval_points(
        self,
        func: Callable[[np.ndarray], float],
        points: Iterable[np.ndarray],
        executor: ProcessPoolExecutor | None,
    ) -> list[float]:
        if executor is None:
            return [func(p) for p in points]
        futures = [executor.submit(func, p) for p in points]
        return [f.result() for f in futures]

    # ------------------------------------------------------------------
    def optimize(
        self,
        objective: Callable[[np.ndarray], float],
        x0: np.ndarray,
        space: DesignSpace,
        constraints: Sequence[Constraint] = (),
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int | None = None,
        parallel: bool = False,
        verbose: bool = False,
    ) -> OptResult:
        if seed is not None:
            np.random.default_rng(seed)
        x0 = self._validate_x0(x0, space)
        self.reset_history()

        n = space.dimension
        step = np.asarray(self.step, dtype=float)
        if step.size == 1:
            step = np.full(n, float(step))
        if step.shape != (n,):
            raise ValueError("step must be scalar or of length equal to dimension")

        penalised = self._make_penalised(objective, space, constraints)

        executor = ProcessPoolExecutor(max_workers=self.n_workers) if parallel else None
        try:
            simplex = [x0]
            for i in range(n):
                pt = x0.copy()
                pt[i] += step[i]
                simplex.append(pt)
            fvals = self._eval_points(penalised, simplex, executor)

            self.record(simplex[np.argmin(fvals)], tag="start")
            best = min(fvals)
            no_improv = 0

            for it in range(max_iter):
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                current_best = fvals[0]
                self.record(simplex[0], tag=str(it))
                if verbose:
                    logger.info("%d | best %.6f", it, current_best)

                if best - current_best > tol:
                    best = current_best
                    no_improv = 0
                else:
                    no_improv += 1
                if no_improv >= self.no_improv_break:
                    break

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]

                # Reflection
                xr = centroid + self.alpha * (centroid - worst)
                fr = self._eval_points(penalised, [xr], executor)[0]

                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                    continue

                if fr < fvals[0]:
                    xe = centroid + self.gamma * (xr - centroid)
                    fe = self._eval_points(penalised, [xe], executor)[0]
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    continue

                if fvals[-2] <= fr < fvals[-1]:
                    xoc = centroid + self.beta * (xr - centroid)
                    foc = self._eval_points(penalised, [xoc], executor)[0]
                    if foc <= fr:
                        simplex[-1] = xoc
                        fvals[-1] = foc
                        continue

                xic = centroid + self.delta * (worst - centroid)
                fic = self._eval_points(penalised, [xic], executor)[0]
                if fic < fvals[-1]:
                    simplex[-1] = xic
                    fvals[-1] = fic
                    continue

                new_points = [simplex[0]]
                for p in simplex[1:]:
                    new_points.append(simplex[0] + self.sigma * (p - simplex[0]))
                new_f = self._eval_points(penalised, new_points[1:], executor)
                simplex = new_points
                fvals = [fvals[0]] + list(new_f)
        finally:
            if executor is not None:
                executor.shutdown()

        idx = int(np.argmin(fvals))
        best_x = simplex[idx]
        best_f = fvals[idx]
        return OptResult(best_x=best_x, best_f=float(best_f), history=self.history)
