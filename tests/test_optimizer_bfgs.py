from __future__ import annotations

import numpy as np
import pytest

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
