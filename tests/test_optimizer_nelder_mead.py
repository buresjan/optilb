from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np
import pytest

from optilb import Constraint, DesignSpace, get_objective
from optilb.optimizers import EarlyStopper, NelderMeadOptimizer


_PARALLEL_COUNTER = mp.Value("i", 0)
_PARALLEL_LIMIT = 5


def _reset_parallel_counter() -> None:
    with _PARALLEL_COUNTER.get_lock():
        _PARALLEL_COUNTER.value = 0


def capped_objective(x: np.ndarray) -> float:
    with _PARALLEL_COUNTER.get_lock():
        if _PARALLEL_COUNTER.value >= _PARALLEL_LIMIT:
            raise StopIteration
        _PARALLEL_COUNTER.value += 1
    return float(np.sum(x**2))


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


def test_nm_parallel_nfev() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = NelderMeadOptimizer()
    res = opt.optimize(obj, np.array([1.0, 1.0]), ds, max_iter=20, parallel=True)
    assert res.nfev == opt.nfev
    assert res.nfev >= len(res.history) > 0


def test_nm_parallel_stopiteration_counts() -> None:
    _reset_parallel_counter()
    ds = DesignSpace(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    opt = NelderMeadOptimizer(n_workers=2)
    with pytest.raises(StopIteration):
        opt.optimize(
            capped_objective,
            np.array([0.5, 0.5]),
            ds,
            max_iter=50,
            parallel=True,
            normalize=False,
        )
    assert opt.nfev >= _PARALLEL_COUNTER.value
    assert opt.nfev <= _PARALLEL_COUNTER.value + 1


def test_nm_respects_max_evals() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = NelderMeadOptimizer()
    res = opt.optimize(
        obj,
        np.array([1.0, 1.0]),
        ds,
        max_iter=200,
        max_evals=10,
    )
    assert res.nfev == 10


def test_nm_max_evals_best_point_consistent() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = NelderMeadOptimizer(step=1.0)
    x0 = np.array([-3.0, -3.0])
    res = opt.optimize(obj, x0, ds, max_iter=200, max_evals=2)
    np.testing.assert_allclose(res.best_x, np.array([-2.0, -3.0]))
    assert res.best_f == pytest.approx(np.sum(res.best_x**2), abs=1e-12)
