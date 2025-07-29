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


MAX_ITER = 200


def run_benchmark() -> None:
    import pandas as pd  # local import to avoid hard dependency

    rows: list[dict[str, object]] = []

    objectives = ["quadratic", "rastrigin"]
    dims = [2, 3, 5]

    for obj_name in objectives:
        objective = get_objective(obj_name)
        for dim in dims:
            lower = -5 * np.ones(dim)
            upper = 5 * np.ones(dim)
            space = DesignSpace(lower=lower, upper=upper)
            x0 = np.full(dim, 3.0)
            configs = [
                ("BFGS", BFGSOptimizer(), False),
                ("MADS", MADSOptimizer() if HAS_MADS else None, False),
                ("Nelder-Mead", NelderMeadOptimizer(), False),
                ("Nelder-Mead (parallel)", NelderMeadOptimizer(), True),
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
                res = opt.optimize(
                    objective,
                    x0,
                    space,
                    max_iter=MAX_ITER,
                    parallel=parallel,
                    early_stopper=stopper,
                )
                dt = time.perf_counter() - t0
                rows.append(
                    {
                        "objective": obj_name,
                        "dim": dim,
                        "optimizer": name,
                        "best_f": res.best_f,
                        "evals": len(res.history),
                        "time_s": dt,
                        "early_stop": stopper._counter >= stopper.patience,
                    }
                )

    df = pd.DataFrame(rows)
    df.sort_values(["objective", "dim", "optimizer"], inplace=True)
    print(df.to_markdown(index=False, floatfmt=".3e"))


if __name__ == "__main__":
    run_benchmark()
