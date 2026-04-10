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


def test_bfgs_fd_parallel_batches_full_2d_stencil() -> None:
    ds = DesignSpace(lower=-5 * np.ones(2), upper=5 * np.ones(2))
    x0 = np.array([2.0, 2.0])
    opt = BFGSOptimizer(fd_eps=[1e-3, 2e-3], n_workers=4)

    expected_stencil = {
        (2.001, 2.0),
        (1.999, 2.0),
        (2.0, 2.002),
        (2.0, 1.998),
    }
    stencil_started: set[tuple[float, float]] = set()
    stencil_ready = threading.Event()
    log: list[tuple[str, tuple[float, float]]] = []
    log_lock = threading.Lock()

    def traced_quadratic(x: np.ndarray) -> float:
        point = tuple(float(value) for value in np.round(np.asarray(x, dtype=float), 6))
        with log_lock:
            log.append(("start", point))
            if point in expected_stencil:
                stencil_started.add(point)
                if len(stencil_started) == len(expected_stencil):
                    stencil_ready.set()
        if point in expected_stencil:
            stencil_ready.wait(timeout=0.5)
        else:
            time.sleep(0.01)
        value = quadratic(np.asarray(point, dtype=float))
        with log_lock:
            log.append(("end", point))
        return value

    opt.optimize(
        traced_quadratic,
        x0,
        ds,
        max_iter=1,
        parallel=True,
        normalize=False,
    )

    first_center_end = log.index(("end", (2.0, 2.0)))
    first_stencil_events = log[first_center_end + 1 : first_center_end + 5]
    assert len(first_stencil_events) == 4
    assert all(kind == "start" for kind, _ in first_stencil_events)
    assert {point for _, point in first_stencil_events} == expected_stencil


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
