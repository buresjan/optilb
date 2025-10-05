from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from optilb import DesignSpace
from optilb.optimizers import BFGSOptimizer


def quadratic(x: np.ndarray) -> float:
    return float(np.sum((x - 1.0) ** 2))


def slow_quadratic(x: np.ndarray) -> float:
    time.sleep(0.05)
    return quadratic(x)


def test_bfgs_fd_scalar_eps() -> None:
    ds = DesignSpace(lower=-5 * np.ones(3), upper=5 * np.ones(3))
    x0 = np.array([2.5, -1.0, 0.0])
    opt = BFGSOptimizer(fd_eps=1e-6)
    res = opt.optimize(quadratic, x0, ds, max_iter=200, parallel=False)
    np.testing.assert_allclose(res.best_x, np.ones(3), atol=1e-3)
    assert res.best_f == pytest.approx(0.0, abs=1e-6)


def test_bfgs_fd_array_eps() -> None:
    ds = DesignSpace(lower=-5 * np.ones(2), upper=5 * np.ones(2))
    x0 = np.array([3.0, -2.0])
    opt = BFGSOptimizer(fd_eps=[1e-6, 1e-8])
    res = opt.optimize(quadratic, x0, ds, max_iter=200, parallel=False)
    np.testing.assert_allclose(res.best_x, np.ones(2), atol=1e-3)
    assert res.best_f == pytest.approx(0.0, abs=1e-6)


def test_bfgs_fd_parallel_speed() -> None:
    ds = DesignSpace(lower=-5 * np.ones(6), upper=5 * np.ones(6))
    x0 = np.full(6, 2.0)
    opt = BFGSOptimizer(fd_eps=1e-5, n_workers=6)

    t0 = time.perf_counter()
    opt.optimize(slow_quadratic, x0, ds, max_iter=5, parallel=False)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    opt.optimize(slow_quadratic, x0, ds, max_iter=5, parallel=True)
    t_par = time.perf_counter() - t0

    assert t_par < t_seq


def test_bfgs_parallel_respects_max_evals() -> None:
    ds = DesignSpace(lower=-5 * np.ones(2), upper=5 * np.ones(2))
    x0 = np.array([2.5, -1.5])
    opt = BFGSOptimizer(fd_eps=1e-5, n_workers=4)

    counter = 0
    counter_lock = threading.Lock()

    def counted_quadratic(x: np.ndarray) -> float:
        nonlocal counter
        with counter_lock:
            counter += 1
        time.sleep(0.01)
        return float(np.sum((x - 1.0) ** 2))

    max_evals = 8
    res = opt.optimize(
        counted_quadratic,
        x0,
        ds,
        max_iter=20,
        max_evals=max_evals,
        parallel=True,
    )

    assert res.nfev <= max_evals
    assert counter <= max_evals
    assert opt.last_budget_exhausted is True
