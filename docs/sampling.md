# Latin-Hypercube Sampling (LHS)

`optilb` provides an `lhs` function in `optilb.sampling` to generate
Latin-Hypercube samples within a given `optilb.DesignSpace`. The function uses
SciPy's `qmc.LatinHypercube` sampler under the hood and returns a list of
immutable `DesignPoint` instances.

```python
from optilb import DesignSpace
from optilb.sampling import lhs

ds = DesignSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
points = lhs(10, ds, seed=42)
for p in points:
    print(p.x)
```

Parameters:

- `sample_count` – number of points to draw.
- `design_space` – bounds that the unit-cube sample is scaled to.
- `seed` – forwarded to NumPy's default RNG for reproducibility.
- `centered` – place each point at the centre of its hypercube cell instead of
  relying on the low-discrepancy sampler.
- `scramble` – toggle Owen scrambling when `centered=False`.

When the corresponding bounds are integers, samples are rounded to the nearest
integer so that mixed discrete/continuous design spaces remain valid.
