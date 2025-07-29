# Sampling API

This module exposes the `lhs` function used to create Latin-Hypercube samples.

```python
from optilb import DesignSpace
from optilb.sampling import lhs

ds = DesignSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
points = lhs(10, ds, seed=42)
for p in points:
    print(p.x)
```

See the docstring of `lhs` for full parameter information.
