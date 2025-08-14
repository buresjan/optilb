from __future__ import annotations

import numpy as np

from optilb import DesignSpace, OptimizationProblem, get_objective


def test_problem_bfgs() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    prob = OptimizationProblem(obj, ds, np.array([3.0, 3.0]), optimizer="bfgs")
    res = prob.run()
    np.testing.assert_allclose(res.best_x, np.zeros(2), atol=1e-5)
    assert prob.log is not None
    assert prob.log.nfev == res.nfev
    assert prob.log.runtime >= 0.0


def test_problem_eval_cap() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    prob = OptimizationProblem(
        obj,
        ds,
        np.array([3.0, 3.0]),
        optimizer="nelder-mead",
        max_evals=5,
        max_iter=100,
    )
    res = prob.run()
    assert prob.log is not None and prob.log.early_stopped
    assert res.nfev <= 5
