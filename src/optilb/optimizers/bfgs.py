from __future__ import annotations

import logging
from typing import Callable, Sequence, cast

import numpy as np
from scipy import optimize

from ..core import Constraint, DesignSpace, OptResult
from .base import Optimizer
from .early_stop import EarlyStopper

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
        early_stopper: EarlyStopper | None = None,
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

        jac: Callable[[np.ndarray], np.ndarray] | str | None = self.gradient
        if jac is None:
            for attr in ("grad", "gradient", "jac"):
                maybe = getattr(objective, attr, None)
                if callable(maybe):
                    jac = maybe
                    break
            else:
                jac = "3-point"

        wrapped_obj = self._wrap_objective(objective)

        options: dict[str, float | int | bool] = {
            "maxiter": max_iter,
            "ftol": tol,
            "gtol": tol,
            "disp": verbose,
        }
        if self.step is not None:
            options["eps"] = self.step

        if early_stopper is not None:
            early_stopper.reset()

        def _callback(xk: np.ndarray) -> None:
            self.record(xk, tag=f"{len(self._history)}")
            if early_stopper is not None:
                f_val = wrapped_obj.last_val
                if f_val is None:
                    f_val = float(wrapped_obj(xk))
                if early_stopper.update(f_val):
                    raise StopIteration

        try:
            res = optimize.minimize(  # type: ignore[call-overload]
                cast(Callable[[np.ndarray], float], wrapped_obj),
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
            best_f = wrapped_obj.last_val
            if best_f is None:
                best_f = float(wrapped_obj(best))
            return OptResult(
                best_x=best,
                best_f=float(best_f),
                history=self.history,
                nfev=self.nfev,
            )

        if res.status != 0:
            logger.warning("SciPy optimisation did not converge: %s", res.message)

        return OptResult(
            best_x=res.x,
            best_f=float(res.fun),
            history=self.history,
            nfev=self.nfev,
        )
