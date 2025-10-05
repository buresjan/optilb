# Core API

The core module exports dataclasses that describe the optimisation domain and
results. They are re-exported at package level for convenience.

- `DesignSpace(lower, upper, names=None)` – validates bounds and exposes the
  `dimension` property and read-only `lower`/`upper` arrays.
- `DesignPoint(x, tag=None, timestamp=None)` – records optimisation samples with
  immutable coordinates.
- `Constraint(func, name=None)` – wraps boolean/penalty callbacks and is
  callable.
- `OptResult(best_x, best_f, history=(), nfev=0)` – stores the outcome of an
  optimiser run with a tuple of `DesignPoint` history.
- `OptimizationLog(optimizer, runtime, nfev, early_stopped=False)` – summarises a
  façade run and is attached to `OptimizationProblem.log`.
- `EvaluationBudgetExceeded(max_evals)` – raised when an optimiser reaches the
  evaluation budget.

```python
from optilb import DesignSpace, OptResult

space = DesignSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
result = OptResult(best_x=[0.5, 0.5], best_f=0.42)
print(space.dimension, result.nfev)
```
