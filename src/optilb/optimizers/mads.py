from __future__ import annotations

import logging
from typing import Callable, Sequence

import numpy as np

try:
    import PyNomad
except ImportError:  # pragma: no cover - optional dependency
    PyNomad = None  # type: ignore

from ..core import Constraint, DesignSpace, OptResult
from .base import Optimizer
from .early_stop import EarlyStopper

logger = logging.getLogger("optilb")


class MADSOptimizer(Optimizer):
    """Local optimiser using NOMAD's Mesh Adaptive Direct Search."""

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
        if PyNomad is None:
            raise ImportError(
                "PyNOMAD is not installed; please install PyNomadBBO to use"
                " MADSOptimizer"
            )

        if seed is not None:
            try:
                PyNomad.setSeed(seed)
            except Exception:  # pragma: no cover - sanity
                logger.warning("Failed to set PyNOMAD seed")

        if parallel:
            logger.info("Parallel execution is not yet supported; running serially")

        x0 = self._validate_x0(x0, space)
        objective = self._wrap_objective(objective)
        self.reset_history()
        self.record(x0, tag="start")
        if early_stopper is not None:
            early_stopper.reset()

        dim = space.dimension

        con_funcs: list[Callable[[np.ndarray], float]] = []
        for c in constraints:

            def _wrap(
                func: Callable[[np.ndarray], bool | float],
            ) -> Callable[[np.ndarray], float]:
                def _inner(arr: np.ndarray) -> float:
                    val = func(arr)
                    if isinstance(val, bool):
                        return 0.0 if val else 1.0
                    return float(val)

                return _inner

            con_funcs.append(_wrap(c.func))

        def _bb(point: "PyNomad.PyNomadEvalPoint") -> int:  # type: ignore[name-defined]
            arr = np.array(
                [point.get_coord(i) for i in range(point.size())], dtype=float
            )
            fval = float(objective(arr))
            vals = [fval]
            for g in con_funcs:
                vals.append(float(g(arr)))
            point.setBBO(" ".join(str(v) for v in vals).encode("utf-8"))
            return 1

        output_types = "OBJ" + " PB" * len(con_funcs)
        params = [
            f"DIMENSION {dim}",
            f"MAX_BB_EVAL {max_iter}",
            f"BB_OUTPUT_TYPE {output_types}",
            f"DISPLAY_DEGREE {1 if verbose else 0}",
        ]
        if early_stopper is not None and early_stopper.f_target is not None:
            params.append(f"OBJ_TARGET {early_stopper.f_target}")

        res = PyNomad.optimize(
            _bb,
            x0.tolist(),
            space.lower.tolist(),
            space.upper.tolist(),
            params,
        )
        best = np.array(res["x_best"], dtype=float)
        best_f = float(res["f_best"])
        return OptResult(
            best_x=best,
            best_f=best_f,
            history=self.history,
            nfev=self.nfev,
        )
