from __future__ import annotations

import logging
import os
import threading
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from typing import Callable, Iterable, Iterator, Sequence, cast

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


@contextmanager
def _parallel_executor(
    parallel: bool, n_workers: int | None
) -> Iterator[tuple[Executor | None, bool]]:
    if not parallel:
        yield None, False
        return
    force_threads = os.environ.get("OPTILB_FORCE_THREAD_POOL", "").lower()
    if force_threads in {"1", "true", "yes", "on"}:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            yield executor, False
        return
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            yield executor, True
            return
    except (OSError, PermissionError):
        logger.debug("Falling back to ThreadPoolExecutor for Nelder-Mead")
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        yield executor, False


# ===========================================================================


class NelderMeadOptimizer(Optimizer):
    """(Optionally) parallel Nelder–Mead optimiser.

    Set ``parallel=True`` when calling :meth:`optimize` to evaluate independent
    simplex points concurrently with :class:`concurrent.futures.ProcessPoolExecutor`.
    When the host environment forbids spawning processes (for example in
    sandboxed CI), the optimiser falls back to
    :class:`concurrent.futures.ThreadPoolExecutor` while retaining the same API.

    Set ``normalize=True`` to perform the optimisation in the unit hypercube,
    mapping inputs/outputs accordingly. This makes default step/coefficients
    scale-independent.  Set ``parallel_poll_points=True`` to pre-compute the
    reflection / expansion / contraction candidates each iteration while running
    in parallel. This trades extra objective evaluations for lower iteration
    latency on expensive objectives.
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
        parallel_poll_points: bool = False,
        memoize: bool = False,
    ) -> None:
        super().__init__(memoize=memoize)

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
        self.parallel_poll_points = parallel_poll_points
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
        executor: Executor | None,
        manual_count: bool,
        *,
        map_input: Callable[[np.ndarray], np.ndarray] | None = None,
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
        budget_remaining: int | None = None
        if max_evals is not None:
            remaining = max_evals - self._nfev
            if remaining <= 0:
                self._budget_exhausted = True
                raise EvaluationBudgetExceeded(max_evals)
            budget_remaining = int(remaining)

        # Only handle caching here when we bypass the counting wrapper (process pools).
        use_cache = self._cache_enabled and manual_count
        results: list[float | None] = [None] * len(pts)
        wait_list: list[tuple[int, bytes, threading.Event | None]] = []
        jobs: list[tuple[int, np.ndarray, np.ndarray, bytes | None, threading.Event | None]] = []
        local_results: dict[bytes, float] = {}

        def _map_point(pt: np.ndarray) -> np.ndarray:
            mapped = np.asarray(pt, dtype=float)
            if map_input is not None:
                mapped = np.asarray(map_input(mapped), dtype=float)
            return mapped

        for idx, point in enumerate(pts):
            mapped = _map_point(point)
            key: bytes | None = None
            event: threading.Event | None = None
            cached_val: float | None = None
            should_eval = True
            if use_cache:
                key = self._make_cache_key(mapped)
                try:
                    cached_val, event, should_eval = self._cache_check(
                        key, wait=False
                    )
                except Exception:
                    cached_val = None
                    event = None
                    should_eval = True
                if not should_eval:
                    if cached_val is not None:
                        results[idx] = float(cached_val)
                    else:
                        # Another worker is processing this point; wait after dispatch.
                        wait_list.append((idx, key, event))
                    continue
            if budget_remaining is not None:
                if budget_remaining <= 0:
                    truncated = True
                    with self._state_lock:
                        self._budget_exhausted = True
                    break
                budget_remaining -= 1
                if budget_remaining == 0:
                    with self._state_lock:
                        self._budget_exhausted = True
            jobs.append((idx, point, mapped, key, event))

        def _record_manual(point: np.ndarray, mapped_point: np.ndarray, value: float) -> None:
            if not manual_count:
                return
            arr = np.asarray(point, dtype=float)
            with self._state_lock:
                self._last_eval_point = arr.copy()
            self._record_evaluation(mapped_point, value)

        if executor is None:
            for idx, point, mapped, key, event in jobs:
                if manual_count:
                    with self._state_lock:
                        self._nfev += 1
                try:
                    val = func(point)
                except Exception:
                    if manual_count:
                        with self._state_lock:
                            self._nfev = max(0, self._nfev - 1)
                    if use_cache and key is not None and event is not None:
                        self._cache_fail(key, event)
                    raise
                self._update_best(point, val)
                _record_manual(point, mapped, val)
                if use_cache and key is not None:
                    local_results[key] = float(val)
                    if np.isfinite(val):
                        self._cache_complete(key, val, event)
                    else:
                        self._cache_fail(key, event)
                results[idx] = float(val)
            if truncated:
                assert max_evals is not None
                self._budget_exhausted = True
                raise EvaluationBudgetExceeded(max_evals)
        else:
            if jobs:
                eval_points = [job[1] for job in jobs]
                try:
                    iterator = executor.map(func, eval_points)
                    for (idx, point, mapped, key, event), val in zip(jobs, iterator):
                        if manual_count:
                            with self._state_lock:
                                self._nfev += 1
                        try:
                            self._update_best(point, val)
                            _record_manual(point, mapped, val)
                            if use_cache and key is not None:
                                local_results[key] = float(val)
                                if np.isfinite(val):
                                    self._cache_complete(key, val, event)
                                else:
                                    self._cache_fail(key, event)
                            results[idx] = float(val)
                        except Exception:
                            if manual_count:
                                with self._state_lock:
                                    self._nfev = max(0, self._nfev - 1)
                            if use_cache and key is not None and event is not None:
                                self._cache_fail(key, event)
                            raise
                except StopIteration:
                    raise
                except RuntimeError as exc:
                    if str(exc) == "generator raised StopIteration":
                        raise StopIteration from exc
                    raise

        if use_cache:
            for idx, key, event in wait_list:
                if event is not None:
                    event.wait()
                val: float | None = None
                if key in local_results:
                    val = local_results[key]
                else:
                    val = self._cache_value(key)
                if val is None:
                    raise RuntimeError("Cache bookkeeping error: missing evaluation result")
                results[idx] = float(val)

        if truncated:
            assert max_evals is not None
            self._budget_exhausted = True
            raise EvaluationBudgetExceeded(max_evals)

        for value in results:
            if value is None:
                raise RuntimeError("Missing evaluation result")

        return [float(v) for v in results]

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

        If *parallel* is ``True``, batches of points are evaluated concurrently
        using up to ``n_workers`` processes. Enable ``parallel_poll_points``
        when constructing the optimiser to speculatively score reflection,
        expansion, and contraction candidates together each iteration. When
        ``normalize=True``, optimisation happens in the unit hypercube and
        results/history are mapped back.
        """
        if seed is not None:
            np.random.default_rng(seed)

        # -------------------- optional normalisation -------------------
        # We create picklable wrappers via top-level helpers + partials.
        normalize_transform: SpaceTransform | None = None
        self._history_transform = None

        eval_map: Callable[[np.ndarray], np.ndarray] | None = None

        if normalize:
            normalize_transform = SpaceTransform(space)
            lower = normalize_transform.lower
            span = normalize_transform.span
            eval_map = normalize_transform.from_unit

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

        raw_objective = objective
        counted_objective = self._wrap_objective(
            raw_objective,
            map_input=eval_map,
        )

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

        penalised_counting = self._make_penalised(
            counted_objective, space, constraints
        )
        penalised_raw = self._make_penalised(raw_objective, space, constraints)

        if early_stopper is not None:
            early_stopper.reset()

        # ----------------------------- executor -------------------------
        simplex: list[np.ndarray] = [x0]
        fvals: list[float] = []
        try:
            with _parallel_executor(parallel, self.n_workers) as (
                executor,
                manual_count,
            ):
                evaluate = penalised_raw if manual_count else penalised_counting
                # initial simplex (N + 1 points)
                for i in range(n):
                    pt = x0.copy()
                    pt[i] += step[i]
                    simplex.append(pt)
                fvals = self._eval_points(
                    evaluate,
                    simplex,
                    executor,
                    manual_count,
                    map_input=eval_map,
                )

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
                    if (
                        parallel
                        and executor is not None
                        and self.parallel_poll_points
                    ):
                        fr, fe, foc, fic = self._eval_points(
                            evaluate,
                            [xr, xe, xoc, xic],
                            executor,
                            manual_count,
                            map_input=eval_map,
                        )
                    else:
                        fr = self._eval_points(
                            evaluate,
                            [xr],
                            executor,
                            manual_count,
                            map_input=eval_map,
                        )[0]
                        fe = foc = fic = None

                    # Decision tree (textbook Nelder–Mead)
                    if fvals[0] <= fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                        continue

                    if fr < fvals[0]:
                        if fe is None:
                            fe = self._eval_points(
                                evaluate,
                                [xe],
                                executor,
                                manual_count,
                                map_input=eval_map,
                            )[0]
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                        continue

                    if fvals[-2] <= fr < fvals[-1]:
                        if foc is None:
                            foc = self._eval_points(
                                evaluate,
                                [xoc],
                                executor,
                                manual_count,
                                map_input=eval_map,
                            )[0]
                        if foc <= fr:
                            simplex[-1] = xoc
                            fvals[-1] = foc
                            continue

                    if fic is None:
                        fic = self._eval_points(
                            evaluate,
                            [xic],
                            executor,
                            manual_count,
                            map_input=eval_map,
                        )[0]
                    if fic < fvals[-1]:
                        simplex[-1] = xic
                        fvals[-1] = fic
                        continue

                    # Shrink
                    new_points = [simplex[0]]
                    for p in simplex[1:]:
                        new_points.append(simplex[0] + self.sigma * (p - simplex[0]))
                    new_f = self._eval_points(
                        evaluate,
                        new_points[1:],
                        executor,
                        manual_count,
                        map_input=eval_map,
                    )
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
            evaluations=self.evaluations,
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
