from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Callable, Sequence, cast

import numpy as np
from scipy import optimize

from ..core import Constraint, DesignSpace, OptResult
from .base import Optimizer
from .early_stop import EarlyStopper

logger = logging.getLogger("optilb")

# NOTE: We intentionally do NOT define nested wrappers for the objective here.
# In parallel mode we use ThreadPoolExecutor and call the optimizer's
# wrapped objective directly to preserve nfev/penalties and avoid pickling issues.


class BFGSOptimizer(Optimizer):
    """Local optimiser using SciPy's L-BFGS-B algorithm.

    Parameters
    ----------
    gradient:
        Optional function returning the gradient of the objective. If not
        provided, numerical differentiation is used.
    step:
        Legacy alias for ``fd_eps``. If ``fd_eps`` is not given, ``step`` is
        used as the finite-difference step for numerical gradients.
    fd_eps:
        Finite-difference step size(s) used when estimating gradients
        numerically. May be a scalar or an array with one entry per design
        dimension. When ``None`` (default), a value of ``1e-6`` is used.
    n_workers:
        Maximum number of worker threads used when ``parallel=True`` during
        numerical gradient evaluation.
    """

    def __init__(
        self,
        gradient: Callable[[np.ndarray], np.ndarray] | None = None,
        *,
        step: float | None = None,
        fd_eps: float | Sequence[float] | None = None,
        n_workers: int | None = None,
    ) -> None:
        super().__init__()
        self.gradient = gradient
        self.step = step
        self.fd_eps = fd_eps
        self.n_workers = n_workers

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

        Parameters match :meth:`Optimizer.optimize`. Only bound constraints are
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

        jac: Callable[[np.ndarray], np.ndarray] | None = self.gradient
        if jac is None:
            for attr in ("grad", "gradient", "jac"):
                maybe = getattr(objective, attr, None)
                if callable(maybe):
                    jac = maybe
                    break

        wrapped_obj = self._wrap_objective(objective)

        options: dict[str, float | int | bool] = {
            "maxiter": max_iter,
            "ftol": tol,
            "gtol": tol,
            "disp": verbose,
        }

        if early_stopper is not None:
            early_stopper.reset()

        def _callback(xk: np.ndarray) -> None:
            self.record(xk, tag=f"{len(self._history)}")
            if early_stopper is not None:
                # Always recompute f(xk) to avoid races with threaded FD evals
                f_val = float(wrapped_obj(xk))
                if early_stopper.update(f_val):
                    raise StopIteration

        # ------------------------------------------------------------------
        # Numerical gradient setup (central differences with bounds handling)
        # ------------------------------------------------------------------
        use_central = False
        _executor = None  # type: ignore[assignment]
        if jac is None:
            fd_eps = self.fd_eps
            if fd_eps is None and self.step is not None:
                fd_eps = self.step

            n = space.dimension
            if fd_eps is None:
                eps_vec = np.full(n, 1e-6, dtype=float)
            else:
                eps_vec = np.asarray(fd_eps, dtype=float)
                if eps_vec.ndim == 0:
                    eps_vec = np.full(n, float(eps_vec), dtype=float)
                elif eps_vec.shape != (n,):
                    raise ValueError("fd_eps must be scalar or match dimension")

            lower = np.asarray(space.lower, dtype=float)
            upper = np.asarray(space.upper, dtype=float)

            def _central_grad(x: np.ndarray) -> np.ndarray:
                x = np.asarray(x, dtype=float)
                n_dim = x.size
                grads = np.empty(n_dim, dtype=float)

                plus_pts: list[np.ndarray] = []
                minus_pts: list[np.ndarray] = []
                h_plus = np.empty(n_dim, dtype=float)
                h_minus = np.empty(n_dim, dtype=float)

                for i in range(n_dim):
                    h = eps_vec[i]
                    hp = min(h, upper[i] - x[i])
                    hm = min(h, x[i] - lower[i])
                    hp = 0.0 if hp < 0.0 else hp
                    hm = 0.0 if hm < 0.0 else hm
                    h_plus[i] = hp
                    h_minus[i] = hm

                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += hp
                    xm[i] -= hm
                    plus_pts.append(xp)
                    minus_pts.append(xm)

                if _executor is not None:
                    f_plus = list(_executor.map(wrapped_obj, plus_pts))
                    f_minus = list(_executor.map(wrapped_obj, minus_pts))
                else:
                    f_plus = [float(wrapped_obj(z)) for z in plus_pts]
                    f_minus = [float(wrapped_obj(z)) for z in minus_pts]

                f0_cache: float | None = None
                for i in range(n_dim):
                    hp = h_plus[i]
                    hm = h_minus[i]
                    if hp > 0.0 and hm > 0.0:
                        grads[i] = (f_plus[i] - f_minus[i]) / (hp + hm)
                    elif hp > 0.0:
                        if f0_cache is None:
                            f0_cache = float(wrapped_obj(x))
                        grads[i] = (f_plus[i] - f0_cache) / hp
                    elif hm > 0.0:
                        if f0_cache is None:
                            f0_cache = float(wrapped_obj(x))
                        grads[i] = (f0_cache - f_minus[i]) / hm
                    else:
                        grads[i] = 0.0
                return grads

            jac = _central_grad
            use_central = True

        # ------------------------- optimisation --------------------------
        try:
            # Decide worker count: default to all cores when parallel+central FD
            _need_pool = bool(parallel and use_central)
            _workers = (self.n_workers or os.cpu_count() or 1) if _need_pool else None
            with (
                ThreadPoolExecutor(max_workers=_workers) if _workers else nullcontext()
            ) as _executor:
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
