from __future__ import annotations

import numpy as np

from optilb import DesignSpace, OptResult
from optilb.optimizers import Optimizer


class DummyOptimizer(Optimizer):
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
