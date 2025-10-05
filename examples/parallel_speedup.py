from __future__ import annotations

import time

import numpy as np

from optilb import DesignSpace
from optilb.objectives import lbm_stub
from optilb.optimizers import NelderMeadOptimizer


def slow_obj(x: np.ndarray) -> float:
    """Slow objective used to demonstrate parallel speed-up."""

    return lbm_stub(x, sleep_ms=50)


def run_demo() -> None:
    """Compare sequential and parallel Nelder-Mead on an expensive objective."""

    ds = DesignSpace(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
    x0 = np.array([0.5, 0.5])

    opt = NelderMeadOptimizer(n_workers=8)

    t0 = time.perf_counter()
    opt.optimize(slow_obj, x0, ds, max_iter=30, parallel=False, normalize=True)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    opt.optimize(slow_obj, x0, ds, max_iter=30, parallel=True, normalize=True)
    t_par = time.perf_counter() - t0

    print(f"Sequential time: {t_seq:.2f} s")
    print(f"Parallel time:   {t_par:.2f} s")
    if t_par > 0:
        print(f"Speed-up: {t_seq / t_par:.2f}x")


if __name__ == "__main__":
    run_demo()
