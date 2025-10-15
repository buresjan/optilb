from __future__ import annotations

import time
import warnings

import numpy as np

from optilb import DesignSpace, get_objective
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


np.random.seed(42)


MAX_EVALS = 200


# Known global minima of the benchmark objectives (noise-free).
OPTIMUM_VALUES = {
    "quadratic": 0.0,
    "rastrigin": 0.0,
    "checkerboard": -1.0,
    "step_rastrigin": 0.0,
    "spiky_sine": -1.5,
}


def run_benchmark() -> None:
    import pandas as pd  # local import to avoid hard dependency

    rows: list[dict[str, object]] = []

    objectives = [
        "quadratic",
        "rastrigin",
        "checkerboard",
        "step_rastrigin",
        "spiky_sine",
    ]
    dims = [2, 3, 5]

    noise_free_kwargs = {
        "checkerboard": {"sigma": 0.0},
        "step_rastrigin": {"sigma": 0.0},
        "spiky_sine": {"sigma": 0.0},
    }

    for obj_name in objectives:
        objective = get_objective(obj_name, **noise_free_kwargs.get(obj_name, {}))
        for dim in dims:
            lower = -5 * np.ones(dim)
            upper = 5 * np.ones(dim)
            space = DesignSpace(lower=lower, upper=upper)
            x0 = np.full(dim, 3.0)
            configs = [
                ("BFGS", BFGSOptimizer(n_workers=4), False),
                (
                    "MADS",
                    MADSOptimizer(n_workers=4) if HAS_MADS else None,
                    False,
                ),
                ("Nelder-Mead", NelderMeadOptimizer(n_workers=4), False),
                (
                    "Nelder-Mead (parallel)",
                    NelderMeadOptimizer(n_workers=4),
                    True,
                ),
                (
                    "Nelder-Mead (parallel poll)",
                    NelderMeadOptimizer(
                        n_workers=4,
                        parallel_poll_points=True,
                    ),
                    True,
                ),
            ]
            for name, opt, parallel in configs:
                if opt is None:
                    warnings.warn(
                        "PyNomad not found; skipping MADSOptimizer",
                        RuntimeWarning,
                    )
                    continue
                stopper = EarlyStopper(eps=1e-6, patience=15, enabled=True)
                t0 = time.perf_counter()
                try:
                    res = opt.optimize(
                        objective,
                        x0,
                        space,
                        max_iter=MAX_EVALS,
                        max_evals=MAX_EVALS,
                        parallel=parallel,
                        early_stopper=stopper,
                    )
                except ImportError as exc:
                    warnings.warn(str(exc), RuntimeWarning)
                    continue
                dt = time.perf_counter() - t0
                optimum_val = OPTIMUM_VALUES.get(obj_name, float("nan"))
                start_f = float(objective(x0))
                rows.append(
                    {
                        "objective": obj_name,
                        "dim": dim,
                        "start_x": x0.tolist(),
                        "start_f": start_f,
                        "optimizer": name,
                        "best_f": res.best_f,
                        "optimum": optimum_val,
                        "evals": res.nfev,
                        "time_s": dt,
                        "early_stop": stopper.stopped,
                        "budget_stop": res.nfev >= MAX_EVALS,
                    }
                )

    df = pd.DataFrame(rows)
    df.sort_values(["objective", "dim", "optimizer"], inplace=True)
    print(df.to_markdown(index=False, floatfmt=".3e"))


if __name__ == "__main__":
    run_benchmark()
