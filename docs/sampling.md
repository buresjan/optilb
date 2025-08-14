# Latin-Hypercube Sampling (LHS)

`optilb` provides a simple `lhs` function in the `optilb.sampling` package to generate Latin-Hypercube samples within a given `optilb.DesignSpace`.

```python
from optilb import DesignSpace
from optilb.sampling import lhs

ds = DesignSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
points = lhs(10, ds, seed=42)
for p in points:
    print(p.x)
```

The function supports integer bounds, optional scrambling, and centering of points within each hypercube cell.
