# Scheduled Optimisation Runs

The `optilb.runner` module provides utilities to execute an optimiser across
multiple scale levels. Each level can adapt step sizes or mesh parameters for
individual optimisers. `run_with_schedule` forwards all keyword arguments to the
underlying optimiser but overrides `max_iter` with `budget_per_level` so that
iteration budgets are enforced per level. When an `EarlyStopper` is supplied it
is cloned for each level to avoid cross-talk between stages.

```python
from optilb import DesignSpace, get_objective
from optilb.optimizers import NelderMeadOptimizer, EarlyStopper
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
    early_stopper=EarlyStopper(patience=4, eps=1e-4),
)
print(res.best_x, res.best_f, res.nfev)
```

- `nm_step` adjusts the simplex edge length for `NelderMeadOptimizer`.
- `mads_mesh` provides an initial mesh size when the optimiser exposes an
  `initial_mesh` keyword (and is ignored otherwise).
- `bfgs_eps_scale` rescales the finite-difference step (`fd_eps`) when running
  `BFGSOptimizer` so that step sizes shrink across levels.

`run_with_schedule` combines the history from every level and returns a single
`OptResult` with the best design found and total evaluation count.
