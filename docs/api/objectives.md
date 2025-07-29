# Objectives API

This module exposes simple analytic benchmark functions plus a lightweight
``lbm_stub`` surrogate. Use `optilb.get_objective` to obtain them by name.
Available names include ``quadratic``, ``rastrigin``, ``noisy_discontinuous``,
``checkerboard``, ``step_rastrigin``, ``spiky_sine`` and ``lbm_stub``.

```python
from optilb import get_objective
import numpy as np

rastrigin = get_objective("rastrigin")
print(rastrigin(np.zeros(2)))  # 0.0
```
