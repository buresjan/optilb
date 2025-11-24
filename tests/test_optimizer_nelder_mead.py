from __future__ import annotations

import multiprocessing as mp
import time
from concurrent.futures import Executor
from typing import Callable, Iterable

import numpy as np
import pytest

from optilb import Constraint, DesignSpace, get_objective
from optilb.optimizers import EarlyStopper, NelderMeadOptimizer


_PARALLEL_COUNTER = mp.Value("i", 0)
_PARALLEL_LIMIT = 5


def _reset_parallel_counter() -> None:
    with _PARALLEL_COUNTER.get_lock():
        _PARALLEL_COUNTER.value = 0


def capped_objective(x: np.ndarray) -> float:
    with _PARALLEL_COUNTER.get_lock():
        if _PARALLEL_COUNTER.value >= _PARALLEL_LIMIT:
            raise StopIteration
        _PARALLEL_COUNTER.value += 1
    return float(np.sum(x**2))


def slow_objective(x: np.ndarray) -> float:
    time.sleep(0.1)
    return float(np.sum(x**2))


class RecordingNelderMead(NelderMeadOptimizer):
    """Test helper that records batch sizes passed to _eval_points."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.batch_sizes: list[int] = []

    def _eval_points(  # type: ignore[override]
        self,
        func: Callable[[np.ndarray], float],
        points: Iterable[np.ndarray],
        executor: Executor | None,
        manual_count: bool,
    ) -> list[float]:
        pts_list = list(points)
        self.batch_sizes.append(len(pts_list))
        return super()._eval_points(func, pts_list, executor, manual_count)


class RecordingMemoNelderMead(NelderMeadOptimizer):
    """Capture _eval_points batches for memoisation checks."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.recorded_batches: list[list[np.ndarray]] = []

    def _eval_points(  # type: ignore[override]
        self,
        func: Callable[[np.ndarray], float],
        points: Iterable[np.ndarray],
        executor: Executor | None,
        manual_count: bool,
        *,
        map_input: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> list[float]:
        pts_list = [np.asarray(p, dtype=float) for p in points]
        self.recorded_batches.append(pts_list)
        return super()._eval_points(
            func,
            pts_list,
            executor,
            manual_count,
            map_input=map_input,
        )


def test_nm_parallel_poll_points_batches() -> None:
    ds = DesignSpace(lower=-5 * np.ones(2), upper=5 * np.ones(2))
    x0 = np.array([1.0, 1.0])
    obj = get_objective("quadratic")

    baseline = RecordingNelderMead(n_workers=2)
    res_base = baseline.optimize(obj, x0, ds, max_iter=2, parallel=True)
    assert 4 not in baseline.batch_sizes

    speculative = RecordingNelderMead(n_workers=2, parallel_poll_points=True)
    res_spec = speculative.optimize(obj, x0, ds, max_iter=2, parallel=True)
    assert 4 in speculative.batch_sizes

    np.testing.assert_allclose(res_base.best_x, res_spec.best_x, atol=1e-6)
    assert res_base.best_f == pytest.approx(res_spec.best_f, abs=1e-12)


def test_nm_quadratic_dims() -> None:
    for dim in (2, 5):
        ds = DesignSpace(lower=-5 * np.ones(dim), upper=5 * np.ones(dim))
        x0 = np.full(dim, 3.0)
        obj = get_objective("quadratic")
        opt = NelderMeadOptimizer()
        res = opt.optimize(obj, x0, ds, max_iter=200)
        np.testing.assert_allclose(res.best_x, np.zeros(dim), atol=1e-3)
        assert res.best_f == pytest.approx(0.0, abs=1e-6)
        assert len(res.history) >= 1


def test_nm_plateau_cliff_constraint() -> None:
    ds = DesignSpace(lower=[-2.0], upper=[2.0])
    obj = get_objective("plateau_cliff")
    cons = [Constraint(func=lambda x: x[0] - 0.0, name="x<=0")]
    opt = NelderMeadOptimizer()
    res = opt.optimize(obj, np.array([-1.0]), ds, constraints=cons, max_iter=100)
    assert res.best_x[0] <= 0.0
    assert res.best_f == pytest.approx(1.0, abs=1e-6)


def test_nm_parallel() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = NelderMeadOptimizer()
    res = opt.optimize(obj, np.array([1.0, 1.0]), ds, max_iter=50, parallel=True)
    np.testing.assert_allclose(res.best_x, np.zeros(2), atol=1e-3)
    assert res.best_f == pytest.approx(0.0, abs=1e-6)


def test_nm_parallel_speed() -> None:
    ds = DesignSpace(lower=-1 * np.ones(4), upper=np.ones(4))
    x0 = np.full(4, 0.5)

    opt = NelderMeadOptimizer()
    t0 = time.perf_counter()
    opt.optimize(slow_objective, x0, ds, max_iter=1, parallel=False)
    t_seq = time.perf_counter() - t0
    t0 = time.perf_counter()
    opt.optimize(slow_objective, x0, ds, max_iter=1, parallel=True)
    t_par = time.perf_counter() - t0
    assert t_par <= t_seq


def test_nm_early_stop_plateau() -> None:
    ds = DesignSpace(lower=[-2.0], upper=[2.0])
    obj = get_objective("plateau_cliff")
    stopper = EarlyStopper(eps=0.0, patience=5)
    opt = NelderMeadOptimizer()
    res = opt.optimize(obj, np.array([0.5]), ds, max_iter=50, early_stopper=stopper)
    assert res.best_f == pytest.approx(0.0, abs=1e-6)
    assert len(res.history) < 50


def test_nm_nfev() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = NelderMeadOptimizer()
    res = opt.optimize(obj, np.array([1.0, 1.0]), ds, max_iter=20)
    assert res.nfev == opt.nfev
    assert res.nfev >= len(res.history)
    assert len(res.evaluations) == res.nfev


def test_nm_parallel_nfev() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = NelderMeadOptimizer()
    res = opt.optimize(obj, np.array([1.0, 1.0]), ds, max_iter=20, parallel=True)
    assert res.nfev == opt.nfev
    assert res.nfev >= len(res.history) > 0
    assert len(res.evaluations) == res.nfev


def test_nm_parallel_stopiteration_counts() -> None:
    _reset_parallel_counter()
    ds = DesignSpace(lower=[-1.0, -1.0], upper=[1.0, 1.0])
    opt = NelderMeadOptimizer(n_workers=2)
    with pytest.raises(StopIteration):
        opt.optimize(
            capped_objective,
            np.array([0.5, 0.5]),
            ds,
            max_iter=50,
            parallel=True,
            normalize=False,
        )
    assert opt.nfev >= _PARALLEL_COUNTER.value
    assert opt.nfev <= _PARALLEL_COUNTER.value + 1


def test_nm_respects_max_evals() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = NelderMeadOptimizer()
    res = opt.optimize(
        obj,
        np.array([1.0, 1.0]),
        ds,
        max_iter=200,
        max_evals=10,
    )
    assert res.nfev == 10
    assert len(res.evaluations) == 10


def test_nm_max_evals_best_point_consistent() -> None:
    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = NelderMeadOptimizer(step=1.0)
    x0 = np.array([-3.0, -3.0])
    res = opt.optimize(obj, x0, ds, max_iter=200, max_evals=2)
    np.testing.assert_allclose(res.best_x, np.array([-2.0, -3.0]))
    assert res.best_f == pytest.approx(np.sum(res.best_x**2), abs=1e-12)


def test_nm_normalized_evaluations_in_original_space() -> None:
    ds = DesignSpace(
        lower=np.array([5.0, 10.0]),
        upper=np.array([6.0, 12.0]),
    )
    obj = get_objective("quadratic")
    opt = NelderMeadOptimizer()
    res = opt.optimize(
        obj,
        np.array([5.5, 11.0]),
        ds,
        max_iter=10,
        normalize=True,
    )
    assert len(res.evaluations) == res.nfev
    for record in res.evaluations:
        assert np.all(record.x >= ds.lower) and np.all(record.x <= ds.upper)


def test_nm_memoize_reuses_values() -> None:
    ds = DesignSpace(lower=np.array([-1.0]), upper=np.array([1.0]))
    x0 = np.array([0.25])

    call_counter = {"count": 0}

    def sleepy(x: np.ndarray) -> float:
        call_counter["count"] += 1
        return float(np.sum(x**2))

    opt_plain = NelderMeadOptimizer(step=0.0)
    res_plain = opt_plain.optimize(sleepy, x0, ds, max_iter=0, normalize=False)
    calls_plain = call_counter["count"]

    call_counter["count"] = 0
    opt_cached = NelderMeadOptimizer(step=0.0, memoize=True)
    res_cached = opt_cached.optimize(sleepy, x0, ds, max_iter=0, normalize=False)

    assert call_counter["count"] < calls_plain
    assert res_cached.best_f == pytest.approx(res_plain.best_f)
    assert res_cached.nfev <= res_plain.nfev


def test_nm_parallel_memoize_handles_duplicate_simplex() -> None:
    ds = DesignSpace(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
    x0 = np.array([0.3, -0.2])

    opt = NelderMeadOptimizer(step=0.0, memoize=True, n_workers=2)
    res = opt.optimize(
        get_objective("quadratic"),
        x0,
        ds,
        max_iter=0,
        parallel=True,
        normalize=False,
    )

    assert res.nfev == 1
    assert len(res.evaluations) == 1
    assert res.best_f == pytest.approx(float(np.sum(x0**2)), abs=1e-12)
    assert len(opt._cache) == 1


def test_nm_parallel_poll_points_respects_memoize() -> None:
    ds = DesignSpace(lower=np.array([-1.0, -1.0]), upper=np.array([1.0, 1.0]))
    x0 = np.array([0.3, -0.15])

    opt = RecordingMemoNelderMead(
        memoize=True,
        n_workers=2,
        parallel_poll_points=True,
        alpha=0.0,
    )
    res = opt.optimize(
        get_objective("quadratic"),
        x0,
        ds,
        max_iter=1,
        parallel=True,
        normalize=False,
    )

    # ensure speculative batch included duplicates (reflection/expansion/outside contraction coincide)
    duplicate_batches = []
    for batch in opt.recorded_batches:
        if len(batch) != 4:
            continue
        rounded = [tuple(np.round(pt, 12)) for pt in batch]
        if len(set(rounded)) < len(rounded):
            duplicate_batches.append(batch)
    assert duplicate_batches, "Expected speculative poll batch to contain duplicates"

    # Memoisation should avoid re-evaluating duplicate vertices.
    assert res.nfev == len(opt._cache) == 5
    assert res.best_f <= float(np.sum(x0**2))
    # Cache must contain entries for all unique vertices encountered.
    assert len(opt._cache) == 5


def test_nm_uses_precomputed_simplex_with_normalization() -> None:
    ds = DesignSpace(
        lower=np.array([10.0, -5.0]),
        upper=np.array([20.0, 5.0]),
    )
    simplex = [
        np.array([10.0, -5.0]),
        np.array([12.0, -3.0]),
        np.array([15.0, 0.0]),
    ]
    simplex_values = [float(np.sum(pt**2)) for pt in simplex]

    def fail_objective(_: np.ndarray) -> float:
        raise AssertionError("Objective should not be called for initial simplex")

    opt = NelderMeadOptimizer()
    res = opt.optimize(
        fail_objective,
        np.array([12.0, -3.0]),
        ds,
        initial_simplex=simplex,
        initial_simplex_values=simplex_values,
        max_iter=0,
        normalize=True,
    )

    best_idx = int(np.argmin(simplex_values))
    np.testing.assert_allclose(res.best_x, simplex[best_idx])
    assert res.best_f == pytest.approx(simplex_values[best_idx], abs=1e-12)
    assert res.nfev == len(simplex)
    assert len(res.evaluations) == len(simplex)
    for record, expected in zip(res.evaluations, simplex):
        np.testing.assert_allclose(record.x, expected)


def test_nm_precomputed_simplex_dimension_mismatch_raises() -> None:
    ds = DesignSpace(lower=np.zeros(2), upper=np.ones(2))
    simplex = [np.array([0.0]), np.array([0.1]), np.array([0.2])]

    opt = NelderMeadOptimizer()
    with pytest.raises(ValueError, match="dimension"):
        opt.optimize(
            get_objective("quadratic"),
            np.zeros(2),
            ds,
            initial_simplex=simplex,
            initial_simplex_values=[0.0, 0.01, 0.04],
        )
