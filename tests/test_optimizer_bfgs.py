from __future__ import annotations

import numpy as np
import pytest
import optilb.optimizers.bfgs as bfgs_module

from optilb import DesignSpace, get_objective
from optilb.optimizers import BFGSOptimizer, EarlyStopper


def test_bfgs_quadratic_dims() -> None:
    for dim in (2, 5):
        ds = DesignSpace(lower=-5 * np.ones(dim), upper=5 * np.ones(dim))
        x0 = np.full(dim, 3.0)
        obj = get_objective("quadratic")
        opt = BFGSOptimizer()
        res = opt.optimize(obj, x0, ds, max_iter=50)
        np.testing.assert_allclose(res.best_x, np.zeros(dim), atol=1e-5)
        assert res.best_f == pytest.approx(0.0, abs=1e-8)
        assert len(res.history) >= 1
        assert len(res.evaluations) == res.nfev


def test_bfgs_rastrigin_origin() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("rastrigin")
    opt = BFGSOptimizer()
    res = opt.optimize(obj, np.zeros(2), ds, max_iter=100)
    np.testing.assert_allclose(res.best_x, np.zeros(2), atol=1e-5)
    assert res.best_f == pytest.approx(0.0, abs=1e-6)


def test_bfgs_early_stop_target() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    stopper = EarlyStopper(f_target=0.5)
    opt = BFGSOptimizer()
    res = opt.optimize(
        obj, np.array([3.0, 3.0]), ds, early_stopper=stopper, max_iter=100
    )
    assert res.best_f <= 0.5
    assert len(res.history) < 100


def test_bfgs_nfev() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = BFGSOptimizer()
    res = opt.optimize(obj, np.array([1.0, 1.0]), ds, max_iter=10)
    assert res.nfev == opt.nfev
    assert res.nfev >= len(res.history)
    assert len(res.evaluations) == res.nfev


def test_bfgs_respects_max_evals() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = BFGSOptimizer()
    res = opt.optimize(
        obj,
        np.array([3.0, -2.0]),
        ds,
        max_iter=200,
        max_evals=15,
    )
    assert res.nfev <= 15
    assert len(res.evaluations) == res.nfev


def test_bfgs_zero_max_evals() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    x0 = np.array([1.0, -1.0])
    opt = BFGSOptimizer()
    res = opt.optimize(obj, x0, ds, max_iter=20, max_evals=0)
    assert res.nfev == 0
    np.testing.assert_allclose(res.best_x, x0)
    assert opt.last_budget_exhausted is True
    assert len(res.evaluations) == 0


def test_bfgs_budget_returns_best_evaluation() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    opt = BFGSOptimizer(fd_eps=1.0)
    x0 = np.array([0.0, 0.0])

    res = opt.optimize(
        lambda x: float(-x[0]),
        x0,
        ds,
        max_iter=100,
        max_evals=2,
    )

    best_record = min(res.evaluations, key=lambda record: record.value)
    assert res.best_f == pytest.approx(best_record.value, abs=1e-12)
    np.testing.assert_allclose(res.best_x, best_record.x)


def test_bfgs_stopiteration_returns_best_evaluation(monkeypatch: pytest.MonkeyPatch) -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    opt = BFGSOptimizer(fd_eps=1.0)
    x0 = np.array([0.0, 0.0])

    def fake_minimize(
        fun, x0_input, method=None, jac=None, bounds=None, callback=None, options=None
    ):  # type: ignore[no-untyped-def]
        x_start = np.asarray(x0_input, dtype=float)
        x_better = x_start.copy()
        if bounds is None:
            raise AssertionError("bounds must be provided")
        x_better[0] = float(bounds[0][1])
        fun(x_start)
        fun(x_better)
        raise StopIteration

    monkeypatch.setattr(bfgs_module.optimize, "minimize", fake_minimize)

    res = opt.optimize(
        lambda x: float(-x[0]),
        x0,
        ds,
        max_iter=100,
    )

    best_record = min(res.evaluations, key=lambda record: record.value)
    assert res.best_f == pytest.approx(best_record.value, abs=1e-12)
    np.testing.assert_allclose(res.best_x, best_record.x)
