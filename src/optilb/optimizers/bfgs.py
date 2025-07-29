from __future__ import annotations

from typing import Callable, Sequence

import logging
import numpy as np
from scipy import optimize

from ..core import Constraint, DesignSpace, OptResult
from .base import Optimizer

logger = logging.getLogger("optilb")


class BFGSOptimizer(Optimizer):
    """Local optimiser using SciPy's L-BFGS-B algorithm.

    Parameters
    ----------
    gradient:
        Optional function returning the gradient of the objective.  If not
        provided, numerical differentiation is used.
    step:
        Optional finite-difference step used when estimating gradients
        numerically.
    """

    def __init__(
        self,
        gradient: Callable[[np.ndarray], np.ndarray] | None = None,
        *,
        step: float | None = None,
    ) -> None:
        super().__init__()
        self.gradient = gradient
        self.step = step

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
        """Run the optimiser.

        Parameters match :meth:`Optimizer.optimize`.  Only bound constraints are
        enforced.
        """
        if seed is not None:
            np.random.default_rng(seed)  # for API symmetry; not used directly
        x0 = self._validate_x0(x0, space)
        if constraints:
            logger.warning(
                "BFGSOptimizer ignores nonlinear constraints; only bounds are enforced"
            )
        self.reset_history()
        self.record(x0, tag="start")

        bounds = list(zip(space.lower, space.upper))

        jac = self.gradient
        if jac is None:
            for attr in ("grad", "gradient", "jac"):
                maybe = getattr(objective, attr, None)
                if callable(maybe):
                    jac = maybe
                    break
            else:
                jac = "3-point"

        options: dict[str, float | int | bool] = {
            "maxiter": max_iter,
            "ftol": tol,
            "gtol": tol,
            "disp": verbose,
        }
        if self.step is not None:
            options["eps"] = self.step

        def _callback(xk: np.ndarray) -> None:
            self.record(xk, tag=f"{len(self._history)}")

        try:
            res = optimize.minimize(
                objective,
                x0,
                method="L-BFGS-B",
                jac=jac,
                bounds=bounds,
                callback=_callback,
                options=options,
            )
        except StopIteration:
            logger.info("Optimization stopped early by callback")
            best = self.history[-1].x
            return OptResult(
                best_x=best, best_f=float(objective(best)), history=self.history
            )

        if res.status != 0:
            logger.warning("SciPy optimisation did not converge: %s", res.message)

        return OptResult(best_x=res.x, best_f=float(res.fun), history=self.history)
