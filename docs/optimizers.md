# Local Optimizers

`optilb.optimizers` bundles several local search algorithms that share a common
`Optimizer` base class. Each optimiser records every evaluated design point and
returns an `OptResult` with the full history, a complete evaluation log, and the
evaluation count (`nfev`). A shared `max_evals` budget is tracked inside the
base class; when the budget is
reached an `EvaluationBudgetExceeded` exception is raised, the best-known point
is recorded, and façade helpers report the run as early-stopped.

Common keyword arguments supported by most optimisers:

- `max_iter` – soft iteration limit (solver specific).
- `max_evals` – hard evaluation budget (enforced by the base class).
- `parallel` – enable multi-threaded or multi-process execution where supported.
- `normalize` – operate in the unit hypercube and map results back afterwards.
- `memoize` – cache completed evaluations (per optimiser instance) to avoid
  re-running the objective when the same point recurs. Disabled by default and
  ignored when caching is not supported (for example, MADS with PyNomad).
- `early_stopper` – an `EarlyStopper` instance with `eps`, `patience`,
  `f_target`, `time_limit`, and `enabled` controls. `update()` is called after
  each accepted iterate; when it returns `True` the run terminates early and the
  stopper exposes `stopped=True`.

```python
from optilb import DesignSpace, get_objective
from optilb.optimizers import BFGSOptimizer, EarlyStopper

ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
obj = get_objective("quadratic")
opt = BFGSOptimizer(n_workers=2)
res = opt.optimize(
    obj,
    ds.lower,
    ds,
    normalize=True,
    max_evals=150,
    early_stopper=EarlyStopper(patience=6, eps=1e-4),
    parallel=True,
)
print(res.best_x, res.best_f, res.nfev)
print(len(res.evaluations))
```

Built-in optimisers
-------------------

- `BFGSOptimizer` – wraps SciPy's L-BFGS-B. When `normalize=True` (default) it
  works in `[0, 1]^d` using a `SpaceTransform`, records history and evaluations in original
  coordinates, and supports numerical gradients via central differences. Use
  `fd_eps` (or the legacy alias `step`) to set finite-difference steps; pass
  `n_workers` to parallelise gradient evaluations with threads when
  `parallel=True`. Set `memoize=True` to reuse repeated evaluations during
  central-difference sweeps.
- `NelderMeadOptimizer` – derivative-free simplex search with optional
  normalisation and process-based parallelism. Objectives and constraints must be
  picklable when running with `parallel=True`. Set `parallel_poll_points=True` to
  speculatively evaluate reflection / expansion / contraction candidates each
  iteration (trading extra objective calls for lower latency). The optimiser
  resamples the simplex after each iteration, honours constraint callbacks by
  applying the configured penalty, and captures every evaluated simplex vertex
  (even when running in parallel processes). With `memoize=True`, duplicate
  simplex vertices are short-circuited across sequential, threaded, and process
  pools, including speculative batches triggered by `parallel_poll_points=True`.
  Pass both `initial_simplex` and `initial_simplex_values` to start from a
  pre-evaluated simplex (dimension + 1 vertices). Vertices are supplied in
  physical coordinates and are mapped to the unit cube automatically when
  `normalize=True`. When provided, the initial simplex evaluation phase is
  skipped and Nelder-Mead proceeds directly to simplex transformations. Bounds
  and constraints are checked; invalid vertices receive the penalty value.

### Nelder–Mead with a predefined simplex

```python
import numpy as np
from optilb import DesignSpace, get_objective
from optilb.optimizers import NelderMeadOptimizer

space = DesignSpace(lower=np.array([0.0, -5.0]), upper=np.array([10.0, 5.0]))
objective = get_objective("quadratic")

simplex = [
    np.array([0.0, -5.0]),
    np.array([2.0, -3.0]),
    np.array([4.0, -1.0]),
]
simplex_values = [objective(v) for v in simplex]

opt = NelderMeadOptimizer()
res = opt.optimize(
    objective,
    x0=simplex[0],
    space=space,
    initial_simplex=simplex,
    initial_simplex_values=simplex_values,
    normalize=True,  # vertices are mapped to the unit cube automatically
    max_iter=50,
)
print(res.best_x, res.best_f, res.nfev)
```
- `MADSOptimizer` – interfaces with NOMAD's Mesh Adaptive Direct Search via the
  `PyNomadBBO` package. Pass `normalize=True` to work in the unit cube (finite,
  non-degenerate bounds required). Provide `n_workers` to limit NOMAD's parallel
  evaluation threads. All evaluations reported by NOMAD are stored in original
  coordinates for post-analysis. Memoisation is currently ignored for this
  optimiser because evaluations are delegated to PyNomad. Normalisation helps
  when objective scaling is wildly unbalanced, but PyNomad's default mesh rules
  may need a higher evaluation budget to match the raw-space solution quality;
  consider increasing `max_iter`/`max_evals` when enabling it.
- `EarlyStopper` – a utility to halt optimisation when progress stalls. Reset it
  between runs (handled automatically by `OptimizationProblem` and
  `run_with_schedule`).

All optimisers expose `history`, `evaluations`, and the `budget_exhausted` flag
on the base class. Use them to inspect the run after calling `optimize`.
