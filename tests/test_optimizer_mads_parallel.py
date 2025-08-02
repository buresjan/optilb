from __future__ import annotations

import time

import numpy as np
import pytest

from optilb import DesignSpace
from optilb.optimizers import MADSOptimizer


def slow_obj(x: np.ndarray) -> float:
    time.sleep(0.25)
    return float(np.sum(x**2))


@pytest.mark.skipif(
    pytest.importorskip("nomad", reason="PyNOMAD not installed") is None,
    reason="PyNOMAD not installed",
)
def test_mads_parallel_speed():
    ds = DesignSpace(lower=-1 * np.ones(3), upper=np.ones(3))
    x0 = np.full(3, 0.5)

    opt = MADSOptimizer(n_workers=4)

    t0 = time.perf_counter()
    opt.optimize(slow_obj, x0, ds, max_iter=60, parallel=False)
    t_seq = time.perf_counter() - t0

    t0 = time.perf_counter()
    opt.optimize(slow_obj, x0, ds, max_iter=60, parallel=True)
    t_par = time.perf_counter() - t0

    # Allow some slack, but expect meaningful improvement for multi-eval steps
    assert (
        t_par < t_seq * 0.8 or (t_seq - t_par) > 2.0
    )  # either 20% faster or >2s saved


def test_mads_api_parallel_flag_doesnt_break_seq():
    ds = DesignSpace(lower=np.array([-2.0, -2.0]), upper=np.array([2.0, 2.0]))
    x0 = np.array([1.0, 1.0])

    opt = MADSOptimizer()
    res_seq = opt.optimize(
        lambda x: float(np.sum(x**2)), x0, ds, max_iter=50, parallel=False
    )
    res_par = opt.optimize(
        lambda x: float(np.sum(x**2)), x0, ds, max_iter=50, parallel=True
    )

    assert res_seq.best_f == pytest.approx(res_par.best_f, rel=1e-6, abs=1e-8)
