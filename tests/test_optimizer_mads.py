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
    ds = DesignSpace(lower=np.array([-2.0]), upper=np.array([2.0]))
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


def test_mads_nfev() -> None:
    ds = DesignSpace(lower=np.array([-2.0]), upper=np.array([2.0]))
    obj = get_objective("quadratic")
    opt = MADSOptimizer()
    res = opt.optimize(obj, np.array([1.0]), ds, max_iter=20)
    assert res.nfev == opt.nfev
    assert res.nfev >= len(res.history)


def test_mads_normalize_smoke() -> None:
    ds = DesignSpace(lower=np.array([-1000.0, -1.0]), upper=np.array([1000.0, 1.0]))
    x0 = np.array([900.0, 0.8])

    def anisotropic(x: np.ndarray) -> float:
        return float(1e6 * x[0] ** 2 + x[1] ** 2)

    opt1 = MADSOptimizer()
    res1 = opt1.optimize(anisotropic, x0, ds, max_iter=40, seed=0)
    opt2 = MADSOptimizer()
    res2 = opt2.optimize(anisotropic, x0, ds, max_iter=40, seed=0, normalize=True)
    assert res2.best_f <= res1.best_f + 1e-8
    assert res2.nfev <= res1.nfev


def test_mads_normalize_bounds_validation() -> None:
    ds_inf = DesignSpace(lower=np.array([-np.inf]), upper=np.array([1.0]))
    obj = get_objective("quadratic")
    opt = MADSOptimizer()
    with pytest.raises(
        ValueError,
        match="normalize=True requires finite, non-degenerate bounds for all variables",
    ):
        opt.optimize(obj, np.array([0.0]), ds_inf, max_iter=5, normalize=True)

    ds_deg = DesignSpace(lower=np.array([0.0]), upper=np.array([0.0]))
    with pytest.raises(
        ValueError,
        match="normalize=True requires finite, non-degenerate bounds for all variables",
    ):
        opt.optimize(obj, np.array([0.0]), ds_deg, max_iter=5, normalize=True)

    ds_nan = DesignSpace(lower=np.array([0.0]), upper=np.array([np.nan]))
    with pytest.raises(
        ValueError,
        match="normalize=True requires finite, non-degenerate bounds for all variables",
    ):
        opt.optimize(obj, np.array([0.0]), ds_nan, max_iter=5, normalize=True)


def test_mads_normalize_mapping() -> None:
    ds = DesignSpace(lower=np.array([-2.0, -4.0]), upper=np.array([2.0, 4.0]))
    seen: list[np.ndarray] = []

    def obj(x: np.ndarray) -> float:
        seen.append(x.copy())
        return float(np.sum(x**2))

    opt = MADSOptimizer()
    x0 = np.array([1.0, 2.0])
    res = opt.optimize(obj, x0, ds, max_iter=20, normalize=True)
    assert res.history[0].tag == "start"
    np.testing.assert_allclose(res.history[0].x, x0)
    assert np.all(res.best_x <= ds.upper) and np.all(res.best_x >= ds.lower)
    for call in seen:
        assert np.all(call <= ds.upper) and np.all(call >= ds.lower)
    assert opt._history_scaled is not None
    assert len(opt._history_scaled) == len(res.history) - 1
    for u in opt._history_scaled:
        assert np.all(u >= 0.0) and np.all(u <= 1.0)

    # ensure scaled history is cleared on subsequent non-normalised runs
    opt.optimize(obj, x0, ds, max_iter=5)
    assert opt._history_scaled is None


def test_mads_constraint_identity_metadata() -> None:
    ds = DesignSpace(lower=np.array([-1.0]), upper=np.array([1.0]))
    obj = get_objective("quadratic")

    def cfunc(x: np.ndarray) -> bool:
        return x[0] <= 0.0

    con = Constraint(func=cfunc, name="c")
    opt = MADSOptimizer()
    opt.optimize(
        obj, np.array([0.5]), ds, constraints=[con], max_iter=10, normalize=True
    )
    assert con.name == "c"
    assert con.func is cfunc
