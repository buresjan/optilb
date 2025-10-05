# Runner API

`optilb.runner` exposes two helpers:

```python
from optilb.runner import ScaleLevel, run_with_schedule
```

- `ScaleLevel` groups per-method scale settings.
- `run_with_schedule` executes an optimiser through successive levels,
  overriding `max_iter` with the per-level budget, rescaling BFGS finite
  difference steps, and cloning any supplied `EarlyStopper`.
- The helper concatenates history from every level and returns a single
  `OptResult`.

Refer to :doc:`../runner` for a complete example.
