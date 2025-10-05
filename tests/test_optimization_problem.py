from __future__ import annotations

import numpy as np

from optilb import DesignSpace, EarlyStopper, OptimizationProblem, get_objective


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


def test_problem_early_stop_logging() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")

    stopper_disabled = EarlyStopper(enabled=False)
    prob_no_stop = OptimizationProblem(
        obj,
        ds,
        np.array([3.0, 3.0]),
        optimizer="bfgs",
        early_stopper=stopper_disabled,
    )
    prob_no_stop.run()
    assert prob_no_stop.log is not None
    assert prob_no_stop.log.early_stopped is False

    stopper_time = EarlyStopper(time_limit=0.0)
    prob_with_stop = OptimizationProblem(
        obj,
        ds,
        np.array([3.0, 3.0]),
        optimizer="bfgs",
        early_stopper=stopper_time,
    )
    prob_with_stop.run()
    assert prob_with_stop.log is not None
    assert prob_with_stop.log.early_stopped is True


def test_problem_eval_cap_history_normalized() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    x0 = np.array([2.0, -1.5])
    prob = OptimizationProblem(
        obj,
        ds,
        x0,
        optimizer="bfgs",
        max_evals=0,
    )
    res = prob.run()
    assert prob.log is not None and prob.log.early_stopped
    assert res.history[0].tag == "start"
    np.testing.assert_allclose(res.history[0].x, x0)
    assert res.history[-1].tag == "cap"
    assert res.nfev == 0
    for pt in res.history:
        assert np.all(pt.x <= ds.upper + 1e-9)
        assert np.all(pt.x >= ds.lower - 1e-9)
