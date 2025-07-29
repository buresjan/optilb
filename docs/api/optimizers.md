# Optimizers API

`optilb.optimizers` defines the common `Optimizer` base class that all local search methods implement.  The toolbox ships a
`BFGSOptimizer` which wraps SciPy's L‑BFGS‑B algorithm for smooth problems.  A
`MADSOptimizer` is also available which interfaces with NOMAD's Mesh Adaptive
Direct Search via the `PyNomadBBO` package.

```python
from optilb.optimizers import Optimizer, BFGSOptimizer

class Dummy(Optimizer):
    def optimize(self, objective, x0, space, constraints=(), **kwargs):
        self.record(x0, tag="start")
        return OptResult(best_x=x0, best_f=objective(x0), history=self.history)
```

Using the provided `BFGSOptimizer`::

    from optilb import DesignSpace, get_objective
    from optilb.optimizers import BFGSOptimizer
    import numpy as np

    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = BFGSOptimizer()
    result = opt.optimize(obj, np.array([3.0, -2.0]), ds)
    print(result.best_x, result.best_f)

Using the `MADSOptimizer` requires the external `PyNomadBBO` package::

    from optilb import DesignSpace, get_objective
    from optilb.optimizers import MADSOptimizer
    import numpy as np

    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = MADSOptimizer()
    result = opt.optimize(obj, np.array([1.0, 1.0]), ds, max_iter=50)
    print(result.best_x, result.best_f)
