from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, cast

import numpy as np

from ..core import DesignPoint, OptResult
from ..optimizers.base import Optimizer
from ..optimizers.early_stop import EarlyStopper


@dataclass
class ScaleLevel:
    """Container for per-method scale settings."""

    nm_step: float | Sequence[float]
    mads_mesh: float
    bfgs_eps_scale: float


def _clone_early_stopper(es: EarlyStopper | None) -> EarlyStopper | None:
    """Return a fresh copy of an :class:`EarlyStopper` instance."""

    if es is None:
        return None
    return type(es)(
        eps=es.eps,
        patience=es.patience,
        f_target=es.f_target,
        time_limit=es.time_limit,
        enabled=es.enabled,
    )


def run_with_schedule(
    optimizer: Optimizer,
    levels: List[ScaleLevel],
    x0: np.ndarray,
    budget_per_level: int,
    **opt_kwargs: Any,
) -> OptResult:
    """Run *optimizer* through successive scale levels.

    Parameters
    ----------
    optimizer:
        Optimiser instance to run.
    levels:
        Ordered list of :class:`ScaleLevel` objects.
    x0:
        Starting point for the first level.
    budget_per_level:
        Maximum number of iterations allowed per level.
    **opt_kwargs:
        Additional keyword arguments forwarded to ``optimizer.optimize``.
        Must include ``objective`` and ``space``.

    Returns
    -------
    OptResult
        Result containing the best design found, accumulated history and total
        evaluation count across all levels.
    """

    from ..optimizers.bfgs import BFGSOptimizer
    from ..optimizers.mads import MADSOptimizer
    from ..optimizers.nelder_mead import NelderMeadOptimizer

    incumbent_x = np.asarray(x0, dtype=float)
    all_history: list[DesignPoint] = []
    total_nfev = 0
    best_x = incumbent_x.copy()
    best_f = float("inf")

    for lvl in levels:
        lvl_kwargs: Dict[str, Any] = dict(opt_kwargs)
        lvl_kwargs["max_iter"] = budget_per_level

        base_eps: float | Sequence[float] | np.ndarray | None = lvl_kwargs.pop(
            "fd_eps", None
        )

        if isinstance(optimizer, NelderMeadOptimizer):
            optimizer.step = lvl.nm_step
        elif isinstance(optimizer, MADSOptimizer):
            if "initial_mesh" in inspect.signature(optimizer.optimize).parameters:
                lvl_kwargs.setdefault("initial_mesh", lvl.mads_mesh)
        elif isinstance(optimizer, BFGSOptimizer):
            if base_eps is None:
                base_eps = 1e-3
            if np.isscalar(base_eps):
                fd_eps: float | np.ndarray = float(cast(float, base_eps)) * float(
                    lvl.bfgs_eps_scale
                )
            else:
                arr = np.asarray(base_eps, dtype=float)
                fd_eps = arr * float(lvl.bfgs_eps_scale)
            optimizer.fd_eps = cast(float | Sequence[float], fd_eps)
            if "fd_eps" in inspect.signature(optimizer.optimize).parameters:
                lvl_kwargs["fd_eps"] = fd_eps

        if "early_stopper" in lvl_kwargs:
            lvl_kwargs["early_stopper"] = _clone_early_stopper(
                lvl_kwargs["early_stopper"]
            )

        res = optimizer.optimize(
            objective=lvl_kwargs.pop("objective"),
            x0=incumbent_x,
            space=lvl_kwargs.pop("space"),
            constraints=lvl_kwargs.pop("constraints", ()),
            **lvl_kwargs,
        )

        all_history.extend(res.history)
        total_nfev += res.nfev
        if res.best_f < best_f:
            best_f = res.best_f
            best_x = res.best_x
        incumbent_x = res.best_x

    return OptResult(
        best_x=best_x,
        best_f=best_f,
        history=tuple(all_history),
        nfev=total_nfev,
    )
