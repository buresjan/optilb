from __future__ import annotations

import numpy as np

from optilb import DesignSpace, get_objective
from optilb.optimizers import BFGSOptimizer, EarlyStopper, NelderMeadOptimizer


def test_quadratic_bfgs_nm() -> None:
    ds = DesignSpace(lower=-5 * np.ones(2), upper=5 * np.ones(2))
    x0 = np.array([3.0, 3.0])
    obj = get_objective("quadratic")

    stopper = EarlyStopper(eps=1e-6, patience=15, enabled=True)
    bfgs = BFGSOptimizer()
    res_bfgs = bfgs.optimize(obj, x0, ds, max_iter=100, early_stopper=stopper)
    assert res_bfgs.best_f < 1e-6

    stopper = EarlyStopper(eps=1e-6, patience=15, enabled=True)
    nm = NelderMeadOptimizer()
    res_nm = nm.optimize(obj, x0, ds, max_iter=200, early_stopper=stopper)
    assert res_nm.best_f < 1e-6
