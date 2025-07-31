from __future__ import annotations

import time

import numpy as np
import pytest

from optilb import Constraint, DesignSpace, get_objective
from optilb.optimizers import EarlyStopper, NelderMeadOptimizer


def slow_objective(x: np.ndarray) -> float:
    time.sleep(0.1)
    return float(np.sum(x**2))


def test_nm_quadratic_dims() -> None:
    for dim in (2, 5):
        ds = DesignSpace(lower=-5 * np.ones(dim), upper=5 * np.ones(dim))
        x0 = np.full(dim, 3.0)
        obj = get_objective("quadratic")
        opt = NelderMeadOptimizer()
        res = opt.optimize(obj, x0, ds, max_iter=200)
        np.testing.assert_allclose(res.best_x, np.zeros(dim), atol=1e-3)
        assert res.best_f == pytest.approx(0.0, abs=1e-6)
        assert len(res.history) >= 1


def test_nm_plateau_cliff_constraint() -> None:
    ds = DesignSpace(lower=[-2.0], upper=[2.0])
    obj = get_objective("plateau_cliff")
    cons = [Constraint(func=lambda x: x[0] - 0.0, name="x<=0")]
    opt = NelderMeadOptimizer()
    res = opt.optimize(obj, np.array([-1.0]), ds, constraints=cons, max_iter=100)
    assert res.best_x[0] <= 0.0
    assert res.best_f == pytest.approx(1.0, abs=1e-6)


def test_nm_parallel() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = NelderMeadOptimizer()
    res = opt.optimize(obj, np.array([1.0, 1.0]), ds, max_iter=50, parallel=True)
    np.testing.assert_allclose(res.best_x, np.zeros(2), atol=1e-3)
    assert res.best_f == pytest.approx(0.0, abs=1e-6)


def test_nm_parallel_speed() -> None:
    ds = DesignSpace(lower=-1 * np.ones(4), upper=np.ones(4))
    x0 = np.full(4, 0.5)

    opt = NelderMeadOptimizer()
    t0 = time.perf_counter()
    opt.optimize(slow_objective, x0, ds, max_iter=1, parallel=False)
    t_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    opt.optimize(slow_objective, x0, ds, max_iter=1, parallel=True)
    t_par = time.perf_counter() - t0
    assert t_par <= t_seq


def test_nm_early_stop_plateau() -> None:
    ds = DesignSpace(lower=[-2.0], upper=[2.0])
    obj = get_objective("plateau_cliff")
    stopper = EarlyStopper(eps=0.0, patience=5)
    opt = NelderMeadOptimizer()
    res = opt.optimize(obj, np.array([0.5]), ds, max_iter=50, early_stopper=stopper)
    assert res.best_f == pytest.approx(0.0, abs=1e-6)
    assert len(res.history) < 50


def test_nm_nfev() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = NelderMeadOptimizer()
    res = opt.optimize(obj, np.array([1.0, 1.0]), ds, max_iter=20)
    assert res.nfev == opt.nfev
    assert res.nfev >= len(res.history)
