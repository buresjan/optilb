from __future__ import annotations

import numpy as np
import pytest

from optilb import (
    DesignSpace,
    OptimizationProblem,
    UnknownOptimizerError,
    get_objective,
)


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


def test_unknown_optimizer() -> None:
    ds = DesignSpace(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    obj = get_objective("quadratic")
    with pytest.raises(UnknownOptimizerError):
        OptimizationProblem(obj, ds, np.array([0.0, 0.0]), optimizer="foo")
