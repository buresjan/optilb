# Optimization Problem

`OptimizationProblem` exposes a uniform façade over the available local
optimisers. It collects the objective, design space and initial point, builds an
optimiser (when given a string alias), forwards shared options, and records a
log for later comparison. This keeps notebook and script code compact while
providing consistent defaults.

```python
import numpy as np
from optilb import DesignSpace, OptimizationProblem, get_objective
from optilb.optimizers import EarlyStopper

space = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
obj = get_objective("quadratic")
problem = OptimizationProblem(
    obj,
    space,
    np.array([3.0, 3.0]),
    optimizer="nelder-mead",  # or an Optimizer instance
    max_evals=250,
    parallel=True,
    normalize=True,
    early_stopper=EarlyStopper(patience=5, eps=1e-4),
    optimizer_options={"n_workers": 2},
    optimize_options={"tol": 1e-5},
)
result = problem.run()
print(result.best_x, result.best_f, problem.log.runtime, problem.log.early_stopped)
```

Key parameters:

- `optimizer` – either an `Optimizer` instance or one of the built-in aliases
  (`"bfgs"`, `"nelder-mead"`, `"mads"`). `optimizer_options` are used only when a
  string alias is supplied.
- `parallel`, `normalize`, `max_iter`, `tol`, `seed`, `max_evals`,
  `early_stopper`, `verbose` – forwarded to the underlying optimiser when
  supported.
- `optimize_options` – merged into the final keyword argument dict so you can
  toggle solver-specific options (e.g. `initial_mesh` for MADS).
- `constraints` – optional sequence of `Constraint` callables. Boolean returns are
  treated as feasibility flags, floats as penalties.

When `max_evals` is set the objective is wrapped in an evaluation cap. Hitting
this cap marks the run as early stopped, records the best design seen so far and
keeps the optimiser history intact. The `OptimizationLog` accessible via
`problem.log` contains `optimizer`, `runtime`, `nfev`, and `early_stopped` fields
for quick diagnostics.
