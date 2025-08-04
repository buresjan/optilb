from __future__ import annotations

import time
import warnings

import numpy as np

from optilb import Constraint, DesignSpace
from optilb.objectives import lbm_stub
from optilb.optimizers import (
    BFGSOptimizer,
    EarlyStopper,
    NelderMeadOptimizer,
)

try:
    from optilb.optimizers import MADSOptimizer

    HAS_MADS = True
except Exception:  # pragma: no cover - optional dependency
    HAS_MADS = False


SLEEP_TIME = 500
MAX_EVALS = 100
OPTIMUM = -2.0


def slow_quadratic(x: np.ndarray) -> float:
    """Quadratic objective with artificial delay."""

    return -lbm_stub(x, sleep_ms=SLEEP_TIME)


def box_constraint(x: np.ndarray) -> float:
    r"""Constraint enforcing :math:`x_i \in [-3, 3]` for all elements."""

    return float(np.max(np.abs(x) - 3.0))


def run_comparison() -> None:
    """Benchmark optimisers on a slow objective from multiple starts."""

    import pandas as pd  # local import to avoid hard dependency

    dim = 2
    lower = -5 * np.ones(dim)
    upper = 5 * np.ones(dim)
    space = DesignSpace(lower=lower, upper=upper)
    constraint = Constraint(box_constraint)

    initial_points = [
        np.array([2.0, 2.0]),
        np.array([-2.0, -2.0]),
        np.array([2.5, -2.5]),
    ]

    configs = [
        ("BFGS", BFGSOptimizer(n_workers=4), False, {}),
        ("BFGS (parallel)", BFGSOptimizer(n_workers=4), True, {}),
        ("Nelder-Mead", NelderMeadOptimizer(), False, {"normalize": False}),
        (
            "Nelder-Mead (parallel)",
            NelderMeadOptimizer(),
            True,
            {"normalize": False},
        ),
        (
            "Nelder-Mead (normalised)",
            NelderMeadOptimizer(),
            False,
            {"normalize": True},
        ),
        (
            "Nelder-Mead (parallel, normalised)",
            NelderMeadOptimizer(),
            True,
            {"normalize": True},
        ),
    ]

    if HAS_MADS:
        configs.extend(
            [
                ("MADS", MADSOptimizer(), False, {}),
                ("MADS (parallel)", MADSOptimizer(), True, {}),
            ]
        )
    else:  # pragma: no cover - optional dependency
        warnings.warn("PyNomad not found; skipping MADSOptimizer", RuntimeWarning)

    rows: list[dict[str, object]] = []

    for idx, x0 in enumerate(initial_points):
        for name, opt, parallel, extra in configs:
            stopper = EarlyStopper(eps=1e-6, patience=10, enabled=True)
            t0 = time.perf_counter()
            res = opt.optimize(
                slow_quadratic,
                x0,
                space,
                constraints=[constraint],
                max_iter=MAX_EVALS,
                parallel=parallel,
                early_stopper=stopper,
                **extra,
            )
            dt = time.perf_counter() - t0
            rows.append(
                {
                    "init_idx": idx,
                    "init_x": x0.tolist(),
                    "optimizer": name,
                    "best_x": res.best_x.tolist(),
                    "best_f": res.best_f,
                    "optimum": OPTIMUM,
                    "evals": res.nfev,
                    "time_s": dt,
                    "early_stop": stopper._counter >= stopper.patience,
                }
            )

    df = pd.DataFrame(rows)
    df.sort_values(["init_idx", "optimizer"], inplace=True)
    print(df.to_markdown(index=False, floatfmt=".3e"))


if __name__ == "__main__":
    run_comparison()
