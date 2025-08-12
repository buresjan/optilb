# Local Optimizers

`optilb` bundles several local search algorithms through the `optilb.optimizers` package. Each optimizer subclasses the common `Optimizer` interface and returns an `OptResult`.

```python
from optilb import DesignSpace, get_objective
from optilb.optimizers import BFGSOptimizer

ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
obj = get_objective("quadratic")
opt = BFGSOptimizer()
res = opt.optimize(obj, ds.lower, ds, normalize=True)
print(res.best_x, res.best_f)
```

Built-in optimizers:

- `BFGSOptimizer` – wraps SciPy's L-BFGS-B for smooth objectives and can
  normalise the design space with `normalize=True`.
- `NelderMeadOptimizer` – supports optional parallel evaluation and normalisation.
- `MADSOptimizer` – binds to NOMAD's Mesh Adaptive Direct Search (requires PyNomadBBO) and can normalise the search space with ``normalize=True`` (requires finite, non-degenerate bounds and reports results in the original coordinates).
- `EarlyStopper` – utility to halt optimisation when progress stalls.

All optimizers accept `n_workers` to control parallel execution where applicable.
