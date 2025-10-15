from __future__ import annotations

import time

import numpy as np

from optilb import DesignSpace
from optilb.objectives import lbm_stub
from optilb.optimizers import NelderMeadOptimizer


def slow_obj(x: np.ndarray) -> float:
    """Slow objective used to demonstrate parallel speed-up."""

    return lbm_stub(x, sleep_ms=5000)


def run_demo() -> None:
    """Compare sequential, parallel, and speculative Nelder-Mead on a slow objective."""

    ds = DesignSpace(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
    x0 = np.array([0.5, 0.5])

    seq_opt = NelderMeadOptimizer(n_workers=8)
    par_opt = NelderMeadOptimizer(n_workers=8)
    spec_opt = NelderMeadOptimizer(n_workers=8, parallel_poll_points=True)

    def timed_run(opt: NelderMeadOptimizer, *, parallel: bool) -> float:
        t0 = time.perf_counter()
        opt.optimize(slow_obj, x0, ds, max_iter=30, parallel=parallel, normalize=True)
        return time.perf_counter() - t0

    t_seq = timed_run(seq_opt, parallel=False)
    t_par = timed_run(par_opt, parallel=True)
    t_spec = timed_run(spec_opt, parallel=True)

    print(f"Sequential time:          {t_seq:.2f} s")
    print(f"Parallel time:            {t_par:.2f} s")
    print(f"Parallel + speculative:   {t_spec:.2f} s")
    if t_par > 0:
        print(f"Parallel speed-up:        {t_seq / t_par:.2f}x")
    if t_spec > 0:
        print(f"Speculative speed-up:     {t_seq / t_spec:.2f}x")


if __name__ == "__main__":
    run_demo()
