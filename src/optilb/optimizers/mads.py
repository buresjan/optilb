from __future__ import annotations

import logging
import os
from typing import Any, Callable, Literal, Sequence

import numpy as np

try:
    import PyNomad
except ImportError:  # pragma: no cover - optional dependency
    PyNomad = None

PYNOMAD_AVAILABLE = PyNomad is not None

from ..core import Constraint, DesignSpace, OptResult
from ..exceptions import MissingDependencyError
from .base import Optimizer
from .early_stop import EarlyStopper

logger = logging.getLogger("optilb")


def _validate_bounds(lb: np.ndarray, ub: np.ndarray) -> None:
    """Ensure bounds are finite and non-degenerate."""

    if not (np.all(np.isfinite(lb)) and np.all(np.isfinite(ub)) and np.all(ub > lb)):
        raise ValueError(
            "normalize=True requires finite, non-degenerate bounds for all variables"
        )


def _to_unit(x: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Map ``x`` from original space to the unit cube."""

    return (np.asarray(x, dtype=float) - lb) / (ub - lb)


def _from_unit(u: np.ndarray, lb: np.ndarray, ub: np.ndarray) -> np.ndarray:
    """Map unit cube ``u`` back to the original space."""

    return lb + np.asarray(u, dtype=float) * (ub - lb)


class MADSOptimizer(Optimizer):
    """Local optimiser using NOMAD's Mesh Adaptive Direct Search.

    Parameters
    ----------
    n_workers:
        Desired number of parallel evaluation threads used by NOMAD when
        ``parallel=True`` in :meth:`optimize`. ``None`` (default) uses all
        available CPU cores.
    """

    def __init__(
        self,
        *,
        n_workers: int | None = None,
        memoize: bool = False,
    ) -> None:
        if memoize:
            logger.warning("MADSOptimizer does not support memoize; disabling cache")
        super().__init__(memoize=False)
        self.n_workers = n_workers

    @classmethod
    def is_available(cls) -> bool:
        """Return whether the underlying PyNomad library is available."""

        return PYNOMAD_AVAILABLE

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
        normalization_mode: Literal["unit"] = "unit",
    ) -> OptResult:
        """Run NOMAD's MADS algorithm.

        Setting ``parallel=True`` lets NOMAD evaluate poll and search trial
        points concurrently using parallel evaluation threads. The number of
        threads is controlled by ``n_workers`` passed to the constructor.

        When ``normalize=True``, the optimiser operates in the unit cube,
        unscaling points before calling the user objective and constraints.
        History and returned results are reported in the original space.

        Parameters
        ----------
        normalize:
            Map the search space to ``[0, 1]^n`` before optimisation. Requires
            finite, non-degenerate bounds. Objective and constraint callbacks
            receive points in the original space; ``best_x`` and ``history`` are
            also reported in the original coordinates.
        normalization_mode:
            Normalisation strategy. Only ``"unit"`` is supported.
        """
        if PyNomad is None:
            raise MissingDependencyError(
                "PyNomad",
                guidance=(
                    "install the optional 'PyNomadBBO' package to enable the MADS"
                    " optimizer (pip install PyNomadBBO)."
                ),
            )

        if seed is not None:
            try:
                PyNomad.setSeed(seed)
            except Exception:  # pragma: no cover - sanity
                logger.warning("Failed to set PyNOMAD seed")
        self._history_scaled = None

        dim = space.dimension
        lower = space.lower.astype(float)
        upper = space.upper.astype(float)
        if normalize:
            if normalization_mode != "unit":
                raise ValueError("Only 'unit' normalization is supported")
            _validate_bounds(lower, upper)

        x0 = self._validate_x0(x0, space)
        self.reset_history()
        self._configure_budget(max_evals)
        self.record(x0, tag="start")
        if early_stopper is not None:
            early_stopper.reset()

        history_scaled: list[np.ndarray] | None = None
        user_objective = objective

        if normalize:
            x0_unit = _to_unit(x0, lower, upper)
            history_scaled = []

            def _unscale(u: np.ndarray) -> np.ndarray:
                return _from_unit(u, lower, upper)

            def objective_unit(u: np.ndarray) -> float:
                return float(user_objective(_unscale(u)))

            space = DesignSpace(np.zeros(dim), np.ones(dim))
            x0 = x0_unit
        else:

            def _unscale(u: np.ndarray) -> np.ndarray:
                return u

            def objective_unit(u: np.ndarray) -> float:
                return float(user_objective(u))

        x0 = self._validate_x0(x0, space)
        objective = self._wrap_objective(objective_unit, map_input=_unscale)

        con_funcs: list[Callable[[np.ndarray], float]] = []
        for c in constraints:

            def _inner(u: np.ndarray, func=c.func) -> float:
                val = func(_unscale(u))
                if isinstance(val, bool):
                    return 0.0 if val else 1.0
                return float(val)

            con_funcs.append(_inner)

        first_eval = True

        def _bb(point: Any) -> int:
            nonlocal first_eval
            arr = np.array(
                [point.get_coord(i) for i in range(point.size())], dtype=float
            )
            x_orig = _unscale(arr)
            if first_eval:
                first_eval = False
            else:
                self.record(x_orig)
                if history_scaled is not None:
                    history_scaled.append(arr.copy())
            fval = float(objective(arr))
            vals = [fval]
            for g in con_funcs:
                vals.append(float(g(arr)))
            point.setBBO(" ".join(str(v) for v in vals).encode("utf-8"))
            return 1

        output_types = "OBJ" + " PB" * len(con_funcs)
        max_bbeval = max_iter if max_evals is None else max_evals

        params = [
            f"DIMENSION {dim}",
            f"MAX_BB_EVAL {max_bbeval}",
            f"BB_OUTPUT_TYPE {output_types}",
            f"DISPLAY_DEGREE {1 if verbose else 0}",
        ]
        if early_stopper is not None and early_stopper.f_target is not None:
            params.append(f"OBJ_TARGET {early_stopper.f_target}")

        if parallel:
            threads = self.n_workers or os.cpu_count() or 1
            params.append(f"NB_THREADS_PARALLEL_EVAL {threads}")

        res = PyNomad.optimize(
            _bb,
            x0.tolist(),
            space.lower.tolist(),
            space.upper.tolist(),
            params,
        )
        best = np.array(res["x_best"], dtype=float)
        if normalize:
            best = _unscale(best)
            self._history_scaled = history_scaled
        else:
            self._history_scaled = None
        best_f = float(res["f_best"])
        result = OptResult(
            best_x=best,
            best_f=best_f,
            history=self.history,
            evaluations=self.evaluations,
            nfev=self.nfev,
        )
        self._clear_budget()
        return result
