# Objectives API

This module exposes simple analytic benchmark functions. Use
`optilb.get_objective` to obtain them by name.

```python
from optilb import get_objective
import numpy as np

rastrigin = get_objective("rastrigin")
print(rastrigin(np.zeros(2)))  # 0.0
```
