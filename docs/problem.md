# Optimization Problem

The `OptimizationProblem` class exposes a uniform façade over the available
local optimisers.  It collects the objective, design space and initial point and
runs the chosen optimiser while tracking run time and evaluation counts.  This
makes side‑by‑side comparisons straightforward.

```python
import numpy as np
from optilb import DesignSpace, OptimizationProblem, get_objective

ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
obj = get_objective("quadratic")
prob = OptimizationProblem(obj, ds, np.array([3.0, 3.0]), optimizer="nelder-mead")
res = prob.run()
print(res.best_x, res.best_f, prob.log.nfev)
```

The log accessible via `prob.log` records the optimizer name, wall‑clock runtime
and whether early stopping was triggered.
