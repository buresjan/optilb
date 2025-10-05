# Analytic Objective Functions

`optilb` ships several toy objective functions useful for quick benchmarking.
They can be obtained via `optilb.get_objective`.

```python
from optilb import get_objective
import numpy as np

quad = get_objective("quadratic")
print(quad(np.array([0.0, 0.0])))  # 0.0
```

Other available names are `rastrigin`, `noisy_discontinuous`, `checkerboard`,
`step_rastrigin`, `spiky_sine`, `plateau_cliff` and `lbm_stub`. The latter is a
purely numerical surrogate for an LBM solver and **does not represent real
physics**.

Factory helpers (`make_noisy_discontinuous`, `make_spiky_sine`,
`make_checkerboard`, `make_step_rastrigin`) accept `sigma` and `seed`
parameters so you can control stochastic behaviour:

```python
from optilb.objectives import make_spiky_sine

obj = make_spiky_sine(sigma=0.0)
```

Use explicit factory functions when you need several independent noisy
instances; `get_objective("spiky_sine", seed=123)` simply forwards keyword
arguments to the underlying factory.
