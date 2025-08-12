from __future__ import annotations

import numpy as np

from optilb.core import DesignSpace
from optilb.optimizers import BFGSOptimizer, MADSOptimizer, NelderMeadOptimizer
from optilb.runner import ScaleLevel, run_with_schedule

levels = [
    ScaleLevel(nm_step=0.2, mads_mesh=0.2, bfgs_eps_scale=0.5),
    ScaleLevel(nm_step=0.1, mads_mesh=0.1, bfgs_eps_scale=0.25),
    ScaleLevel(nm_step=0.05, mads_mesh=0.05, bfgs_eps_scale=0.125),
]


def bumpy(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    rastrigin = 10 * x.size + np.sum(x**2 - 10 * np.cos(2 * np.pi * x))
    bump = 0.05 * np.sin(50 * x[0]) * np.sin(50 * x[1])
    return float(rastrigin + bump)


def _best_per_level(res, obj):
    bests: list[float] = []
    current = None
    for pt in res.history:
        val = obj(pt.x)
        if pt.tag == "start":
            if current is not None:
                bests.append(current)
            current = val
        else:
            if current is None or val < current:
                current = val
    if current is not None:
        bests.append(current)
    return bests


def test_schedule_nelder_mead() -> None:
    ds = DesignSpace(lower=np.zeros(2), upper=np.ones(2))
    x0 = np.array([0.8, 0.9])

    opt = NelderMeadOptimizer()
    res = run_with_schedule(
        opt,
        levels,
        x0,
        budget_per_level=50,
        objective=bumpy,
        space=ds,
        normalize=True,
        seed=0,
    )
    bests = _best_per_level(res, bumpy)
    assert bests[1] <= bests[0]
    assert bests[2] <= bests[1]
    assert opt.step == levels[-1].nm_step

    # expected nfev
    opt_manual = NelderMeadOptimizer()
    inc = x0.copy()
    expected = 0
    for lvl in levels:
        opt_manual.step = lvl.nm_step
        r = opt_manual.optimize(bumpy, inc, ds, max_iter=50, normalize=True, seed=0)
        expected += r.nfev
        inc = r.best_x
    assert res.nfev == expected


def test_schedule_mads() -> None:
    ds = DesignSpace(lower=np.zeros(2), upper=np.ones(2))
    x0 = np.array([0.8, 0.9])

    opt = MADSOptimizer()
    res = run_with_schedule(
        opt,
        levels,
        x0,
        budget_per_level=50,
        objective=bumpy,
        space=ds,
        normalize=True,
        seed=0,
    )
    bests = _best_per_level(res, bumpy)
    assert bests[1] <= bests[0]
    assert bests[2] <= bests[1]

    opt_manual = MADSOptimizer()
    inc = x0.copy()
    expected = 0
    for _lvl in levels:
        r = opt_manual.optimize(bumpy, inc, ds, max_iter=50, normalize=True, seed=0)
        expected += r.nfev
        inc = r.best_x
    assert res.nfev == expected


def test_schedule_bfgs() -> None:
    ds = DesignSpace(lower=np.zeros(2), upper=np.ones(2))
    x0 = np.array([0.8, 0.9])

    opt = BFGSOptimizer()
    res = run_with_schedule(
        opt,
        levels,
        x0,
        budget_per_level=50,
        objective=bumpy,
        space=ds,
        normalize=True,
        seed=0,
    )
    bests = _best_per_level(res, bumpy)
    assert bests[1] <= bests[0]
    assert bests[2] <= bests[1]
    assert np.allclose(np.asarray(opt.fd_eps), 1e-3 * levels[-1].bfgs_eps_scale)

    opt_manual = BFGSOptimizer()
    inc = x0.copy()
    expected = 0
    for lvl in levels:
        opt_manual.fd_eps = 1e-3 * lvl.bfgs_eps_scale
        r = opt_manual.optimize(bumpy, inc, ds, max_iter=50, normalize=True, seed=0)
        expected += r.nfev
        inc = r.best_x
    assert res.nfev == expected
