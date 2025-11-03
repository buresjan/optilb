from __future__ import annotations

import numpy as np

from optilb import DesignSpace
from optilb.optimizers import BFGSOptimizer


def quadratic(x: np.ndarray) -> float:
    return float(np.sum((x - 0.3) ** 2))


def test_unit_space_noop() -> None:
    ds = DesignSpace(lower=np.zeros(3), upper=np.ones(3))
    x0 = np.array([0.8, 0.2, 0.5])
    obj = quadratic

    opt_raw = BFGSOptimizer()
    res_raw = opt_raw.optimize(obj, x0, ds, normalize=False, max_iter=50)

    opt_norm = BFGSOptimizer()
    res_norm = opt_norm.optimize(obj, x0, ds, normalize=True, max_iter=50)

    np.testing.assert_array_equal(res_raw.best_x, res_norm.best_x)
    assert res_raw.best_f == res_norm.best_f
    assert res_raw.nfev == res_norm.nfev
    assert len(res_raw.history) == len(res_norm.history)
    assert len(res_raw.evaluations) == res_raw.nfev
    assert len(res_norm.evaluations) == res_norm.nfev
    for a, b in zip(res_raw.history, res_norm.history):
        np.testing.assert_array_equal(a.x, b.x)
    for a, b in zip(res_raw.evaluations, res_norm.evaluations):
        np.testing.assert_array_equal(a.x, b.x)
        assert a.value == b.value


def test_history_in_original_units() -> None:
    ds = DesignSpace(lower=np.array([5.0, 10.0]), upper=np.array([7.0, 12.0]))
    x0 = np.array([5.5, 10.5])

    def shifted_quad(x: np.ndarray) -> float:
        return float(np.sum((x - np.array([6.0, 11.0])) ** 2))

    opt = BFGSOptimizer()
    res = opt.optimize(shifted_quad, x0, ds, normalize=True, max_iter=20)

    np.testing.assert_array_equal(res.history[0].x, x0)
    for pt in res.history:
        assert np.all(pt.x >= ds.lower) and np.all(pt.x <= ds.upper)
        assert np.all(pt.x > 1.0)
    assert len(res.evaluations) == res.nfev
    for record in res.evaluations:
        assert np.all(record.x >= ds.lower) and np.all(record.x <= ds.upper)
        assert np.all(record.x > 1.0)
