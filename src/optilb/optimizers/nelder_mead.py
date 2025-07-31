from __future__ import annotations

import logging
import sys
from concurrent.futures import ProcessPoolExecutor

# nullcontext is 3.7+.  Provide a tiny shim for 3.6 users.
if sys.version_info >= (3, 7):
    from contextlib import nullcontext  # type: ignore[attr-defined]
else:
    from contextlib import contextmanager

    @contextmanager
    def nullcontext():
        yield


from functools import partial
from typing import Callable, Iterable, Sequence

import numpy as np

from ..core import Constraint, DesignSpace, OptResult
from .base import Optimizer
from .early_stop import EarlyStopper


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
    """Parallel Nelder–Mead optimiser.

    The optimiser can evaluate independent simplex points concurrently when
    :meth:`optimize` is called with ``parallel=True``. Identical numerical
    results are expected in both modes, although parallel execution incurs a
    small process startup overhead.
    """

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

        # NEW: remember worker count so optimize() can pass it to the pool
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
        return list(executor.map(func, points))

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
        early_stopper: EarlyStopper | None = None,
    ) -> OptResult:
        """Run optimisation.

        Parameters
        ----------
        parallel : bool, optional
            Evaluate independent simplex points concurrently using a
            :class:`concurrent.futures.ProcessPoolExecutor`. The objective
            function must be picklable. Numerical results should match the
            sequential mode, but parallel execution may be faster when the
            objective is expensive.
        """
        if seed is not None:
            np.random.default_rng(seed)
        x0 = self._validate_x0(x0, space)
        objective = self._wrap_objective(objective)
        self.reset_history()
        self.record(x0, tag="start")

        n = space.dimension
        step = np.asarray(self.step, dtype=float)
        if step.size == 1:
            step = np.full(n, float(step))
        if step.shape != (n,):
            raise ValueError("step must be scalar or of length equal to dimension")

        penalised = self._make_penalised(objective, space, constraints)

        if early_stopper is not None:
            early_stopper.reset()

        with (
            ProcessPoolExecutor(max_workers=self.n_workers)
            if parallel
            else nullcontext()
        ) as executor:
            simplex = [x0]
            for i in range(n):
                pt = x0.copy()
                pt[i] += step[i]
                simplex.append(pt)
            fvals = self._eval_points(penalised, simplex, executor)

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

                if early_stopper is None:
                    if best - current_best > tol:
                        best = current_best
                        no_improv = 0
                    else:
                        no_improv += 1
                    if no_improv >= self.no_improv_break:
                        break
                else:
                    if early_stopper.update(current_best):
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

        idx = int(np.argmin(fvals))
        best_x = simplex[idx]
        best_f = fvals[idx]
        return OptResult(
            best_x=best_x,
            best_f=float(best_f),
            history=self.history,
            nfev=self.nfev,
        )
