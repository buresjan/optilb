# Analytic Objective Functions

`optilb` ships several toy objective functions useful for quick benchmarking.
They can be obtained via `optilb.get_objective`.

```python
from optilb import get_objective
import numpy as np

quad = get_objective("quadratic")
print(quad(np.array([0.0, 0.0])))  # 0.0
```

Other available names are `rastrigin`, `noisy_discontinuous`, `plateau_cliff`
and `lbm_stub`. The latter is a purely numerical surrogate for an LBM solver
and **does not represent real physics**.
See the docstrings for details on each function.
