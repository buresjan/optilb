from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("PyNomad")  # noqa: E402

from optilb import Constraint, DesignSpace, get_objective  # noqa: E402
from optilb.optimizers import MADSOptimizer  # noqa: E402


def test_mads_quadratic_dims() -> None:
    for dim in (2, 5):
        ds = DesignSpace(lower=-5 * np.ones(dim), upper=5 * np.ones(dim))
        x0 = np.full(dim, 1.0)
        obj = get_objective("quadratic")
        opt = MADSOptimizer()
        res = opt.optimize(obj, x0, ds, max_iter=80)
        np.testing.assert_allclose(res.best_x, np.zeros(dim), atol=1e-2)
        assert res.best_f == pytest.approx(0.0, abs=1e-5)
        assert len(res.history) >= 1


def test_mads_plateau_cliff_constraint() -> None:
    ds = DesignSpace(lower=[-2.0], upper=[2.0])
    obj = get_objective("plateau_cliff")
    cons = [Constraint(func=lambda x: x[0] - 0.0, name="x<=0")]
    opt = MADSOptimizer()
    res = opt.optimize(
        obj,
        np.array([-1.0]),
        ds,
        constraints=cons,
        max_iter=50,
    )
    assert res.best_x[0] <= 0.0
    assert res.best_f == pytest.approx(1.0, abs=1e-6)
