from __future__ import annotations

import numpy as np
import pytest

from optilb import DesignSpace, OptResult
from optilb.optimizers import Optimizer


class DummyOptimizer(Optimizer):
    def __init__(self, *, memoize: bool = False) -> None:
        super().__init__(memoize=memoize)

    def optimize(
        self,
        objective,
        x0: np.ndarray,
        space: DesignSpace,
        constraints=(),
        *,
        max_iter: int = 100,
        tol: float = 1e-6,
        seed: int | None = None,
        parallel: bool = False,
        verbose: bool = False,
        early_stopper=None,
    ) -> OptResult:
        x0 = self._validate_x0(x0, space)
        self.reset_history()
        self.record(x0, tag="start")
        fval = objective(x0)
        return OptResult(best_x=x0, best_f=fval, history=self.history)


def test_dummy_optimizer_history() -> None:
    ds = DesignSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    opt = DummyOptimizer()
    res = opt.optimize(lambda x: float(np.sum(x)), np.zeros(2), ds)
    assert res.best_f == 0.0
    assert len(res.history) == 1
    np.testing.assert_allclose(res.history[0].x, np.zeros(2))


def test_counting_function_memoization() -> None:
    calls = 0

    def obj(x: np.ndarray) -> float:
        nonlocal calls
        calls += 1
        return float(np.sum(x))

    opt = DummyOptimizer(memoize=True)
    wrapped = opt._wrap_objective(obj)
    first = wrapped(np.array([0.5, 0.5]))
    second = wrapped(np.array([0.5, 0.5]))
    assert first == pytest.approx(second)
    assert calls == 1
    assert opt.nfev == 1
