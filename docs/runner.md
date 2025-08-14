# Scheduled Optimisation Runs

The `optilb.runner` module provides utilities to execute an optimiser across
multiple scale levels.  Each level can adapt step sizes or mesh parameters for
individual optimisers.

```python
from optilb import DesignSpace, get_objective
from optilb.optimizers import NelderMeadOptimizer
from optilb.runner import ScaleLevel, run_with_schedule

ds = DesignSpace(lower=[-2.0, -2.0], upper=[2.0, 2.0])
obj = get_objective("quadratic")

levels = [
    ScaleLevel(nm_step=0.5, mads_mesh=0.5, bfgs_eps_scale=1.0),
    ScaleLevel(nm_step=0.2, mads_mesh=0.2, bfgs_eps_scale=0.5),
]

opt = NelderMeadOptimizer()
res = run_with_schedule(
    opt,
    levels,
    x0=[1.0, 1.0],
    budget_per_level=50,
    objective=obj,
    space=ds,
)
print(res.best_x, res.best_f)
```

`run_with_schedule` returns a combined `OptResult` with the best design found and
the accumulated history.
