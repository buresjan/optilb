from __future__ import annotations

import logging
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from functools import partial
from typing import Callable, Iterable, Sequence, cast

import numpy as np

from ..core import Constraint, DesignPoint, DesignSpace, OptResult
from ..exceptions import EvaluationBudgetExceeded
from .base import Optimizer
from .early_stop import EarlyStopper
from .utils import SpaceTransform


# ============================ top-level helpers =============================


def _from_unit(u: np.ndarray, lower: np.ndarray, span: np.ndarray) -> np.ndarray:
    # u in [0,1]^d  -> original space
    return cast(np.ndarray, lower + u * span)


def _objective_from_unit(
    u: np.ndarray,
    objective: Callable[[np.ndarray], float],
    lower: np.ndarray,
    span: np.ndarray,
) -> float:
    # Wrap objective to accept unit-cube input (picklable: top-level fn + partial)
    x = _from_unit(np.asarray(u, dtype=float), lower, span)
    return float(objective(x))


def _constraint_from_unit(
    u: np.ndarray,
    func: Callable[[np.ndarray], bool | float],
    lower: np.ndarray,
    span: np.ndarray,
) -> bool | float:
    # Wrap a single constraint to accept unit-cube input (picklable)
    x = _from_unit(np.asarray(u, dtype=float), lower, span)
    return func(x)


def _evaluate_point(
    x: np.ndarray,
    objective: Callable[[np.ndarray], float],
    lower: np.ndarray,
    upper: np.ndarray,
    constraints: Sequence[Constraint],
    penalty: float,
) -> float:
    """Evaluate *objective* with bound / constraint penalties."""
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


# ===========================================================================


class NelderMeadOptimizer(Optimizer):
    """(Optionally) parallel Nelder–Mead optimiser.

    Set ``parallel=True`` when calling :meth:`optimize` to evaluate independent
    simplex points concurrently with :class:`concurrent.futures.ProcessPoolExecutor`.

    Set ``normalize=True`` to perform the optimisation in the unit hypercube,
    mapping inputs/outputs accordingly. This makes default step/coefficients
    scale-independent.
    """

    # ------------------------------------------------------------------
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

        # remember desired worker count for the executor
        self.n_workers = n_workers
        self._history_transform: SpaceTransform | None = None

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
        """Evaluate *points* either sequentially or in a process pool.

        When running in parallel, the wrapped objective's internal evaluation
        counter (``self._nfev``) lives in a separate process and therefore
        cannot be updated directly.  We manually bump the counter based on the
        number of points evaluated so that ``nfev`` reflects the true cost of
        the optimisation run.
        """

        pts = list(points)
        max_evals = self._max_evals
        truncated = False
        if max_evals is not None:
            remaining = max_evals - self._nfev
            if remaining <= 0:
                self._budget_exhausted = True
                raise EvaluationBudgetExceeded(max_evals)
            if len(pts) > remaining:
                pts = pts[:remaining]
                truncated = True

        if executor is None:
            values: list[float] = []
            for point in pts:
                val = func(point)
                values.append(val)
                self._update_best(point, val)
            if truncated:
                assert max_evals is not None
                self._budget_exhausted = True
                raise EvaluationBudgetExceeded(max_evals)
            return values

        results: list[float] = []
        iterator = executor.map(func, pts)
        try:
            for point, val in zip(pts, iterator):
                results.append(val)
                self._update_best(point, val)
                self._nfev += 1
        except StopIteration:
            raise
        except RuntimeError as exc:
            if str(exc) == "generator raised StopIteration":
                raise StopIteration from exc
            raise

        if truncated:
            assert max_evals is not None
            self._budget_exhausted = True
            raise EvaluationBudgetExceeded(max_evals)
        return results

    # ------------------------------------------------------------------
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
        normalize: bool = False,
    ) -> OptResult:
        """Run Nelder–Mead optimisation.

        If *parallel* is ``True``, reflection, expansion and both contraction
        candidates may be evaluated together each iteration, using up to
        ``n_workers`` processes. When ``normalize=True``, optimisation happens
        in the unit hypercube and results/history are mapped back.
        """
        if seed is not None:
            np.random.default_rng(seed)

        # -------------------- optional normalisation -------------------
        # We create picklable wrappers via top-level helpers + partials.
        normalize_transform: SpaceTransform | None = None
        self._history_transform = None

        if normalize:
            normalize_transform = SpaceTransform(space)
            lower = normalize_transform.lower
            span = normalize_transform.span

            # Wrap objective/constraints for unit space (picklable)
            objective = cast(
                Callable[[np.ndarray], float],
                partial(
                    _objective_from_unit,
                    objective=objective,
                    lower=lower,
                    span=span,
                ),
            )
            constraints = [
                Constraint(
                    func=partial(
                        _constraint_from_unit,
                        func=c.func,
                        lower=lower,
                        span=span,
                    ),
                    name=c.name,
                )
                for c in constraints
            ]

            # Replace design space and initial point with unit versions
            space = DesignSpace(np.zeros(space.dimension), np.ones(space.dimension))
            x0 = normalize_transform.to_unit(x0)

            # keep transform for history post-processing
            self._history_transform = normalize_transform

        # Normal validation/wrapping continues with possibly replaced
        # `space/objective/x0`
        x0 = self._validate_x0(x0, space)
        objective = self._wrap_objective(objective)
        self.reset_history()
        self._configure_budget(max_evals)
        self.record(x0, tag="start")

        n = space.dimension
        step = np.asarray(self.step, dtype=float)
        if step.size == 1:
            step = np.full(n, float(step))
        if step.shape != (n,):
            raise ValueError("step must be scalar or of length equal to dimension")
        # NOTE: do NOT scale `step` by original spans when normalize=True;
        # in unit space, step means exactly that fraction of [0,1].

        penalised = self._make_penalised(objective, space, constraints)

        if early_stopper is not None:
            early_stopper.reset()

        # ----------------------------- executor -------------------------
        simplex: list[np.ndarray] = [x0]
        fvals: list[float] = []
        try:
            with (
                ProcessPoolExecutor(max_workers=self.n_workers)
                if parallel
                else nullcontext()
            ) as executor:
                # initial simplex (N + 1 points)
                for i in range(n):
                    pt = x0.copy()
                    pt[i] += step[i]
                    simplex.append(pt)
                fvals = self._eval_points(penalised, simplex, executor)

                best = min(fvals)
                no_improv = 0

                # --------------------- main loop ---------------------------
                for it in range(max_iter):
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]
                    current_best = fvals[0]

                    self.record(simplex[0], tag=str(it))
                    if verbose:
                        logger.info("%d | best %.6f", it, current_best)

                    # early stopping / no-improve logic
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

                    # Build candidate vertices
                    xr = centroid + self.alpha * (centroid - worst)  # reflection
                    xe = centroid + self.gamma * (xr - centroid)  # expansion
                    xoc = centroid + self.beta * (xr - centroid)  # outside contraction
                    xic = centroid + self.delta * (worst - centroid)  # inside contraction

                    fe: float | None
                    foc: float | None
                    fic: float | None
                    if parallel and executor is not None:
                        fr, fe, foc, fic = self._eval_points(
                            penalised, [xr, xe, xoc, xic], executor
                        )
                    else:
                        fr = self._eval_points(penalised, [xr], executor)[0]
                        fe = foc = fic = None

                    # Decision tree (textbook Nelder–Mead)
                    if fvals[0] <= fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        continue

                    if fr < fvals[0]:
                        if fe is None:
                            fe = self._eval_points(penalised, [xe], executor)[0]
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                        continue

                    if fvals[-2] <= fr < fvals[-1]:
                        if foc is None:
                            foc = self._eval_points(penalised, [xoc], executor)[0]
                        if foc <= fr:
                            simplex[-1] = xoc
                            fvals[-1] = foc
                            continue

                    if fic is None:
                        fic = self._eval_points(penalised, [xic], executor)[0]
                    if fic < fvals[-1]:
                        simplex[-1] = xic
                        fvals[-1] = fic
                        continue

                    # Shrink
                    new_points = [simplex[0]]
                    for p in simplex[1:]:
                        new_points.append(simplex[0] + self.sigma * (p - simplex[0]))
                    new_f = self._eval_points(penalised, new_points[1:], executor)
                    simplex = new_points
                    fvals = [fvals[0]] + list(new_f)
        except EvaluationBudgetExceeded:
            logger.info("Nelder-Mead stopped after reaching the evaluation budget")
        finally:
            self._clear_budget()

        # -------------------------- done -------------------------------
        if fvals:
            idx = int(np.argmin(fvals))
            best_x = simplex[idx]
            best_f = float(fvals[idx])
        else:
            best_eval_point, best_eval_value = self._get_best_evaluation()
            if best_eval_point is not None and best_eval_value is not None:
                best_x = best_eval_point
                best_f = float(best_eval_value)
            else:
                best_x = x0
                best_f = float("nan")

        best_x = np.asarray(best_x, dtype=float)
        # Map result/history back to original coordinates if we normalized
        if normalize and normalize_transform is not None:
            best_x = normalize_transform.from_unit(best_x)
        best_x = np.asarray(best_x, dtype=float).copy()
        self.finalize_history()
        result = OptResult(
            best_x=best_x,
            best_f=float(best_f),
            history=self.history,
            nfev=self.nfev,
        )
        return result

    def finalize_history(self) -> None:
        if self._history_transform is None:
            return
        transform = self._history_transform
        self._history = [
            DesignPoint(
                x=transform.from_unit(pt.x),
                tag=pt.tag,
                timestamp=pt.timestamp,
            )
            for pt in self._history
        ]
        self._history_transform = None
