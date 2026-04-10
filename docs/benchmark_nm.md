# Nelder-Mead Benchmark (SciPy vs optilb)

Use `examples/benchmark_nm_vs_scipy.py` to compare:

- SciPy `optimize.minimize(..., method="Nelder-Mead")`
- `optilb` Nelder-Mead in serial mode
- `optilb` Nelder-Mead in parallel mode

The script is designed to keep the comparison fair:

- same analytic objective function for every run,
- same start points `x0`,
- same initial simplex geometry (`x0 + step * e_i`),
- same Nelder-Mead coefficients (`alpha=1`, `gamma=2`, contractions `0.5`, `sigma=0.5`),
- same stopping budgets (`max_iter`, `max_evals`),
- method-specific extras disabled (`adaptive=False`, `memoize=False`, `normalize=False`,
  `parallel_poll_points=False`).

## Usage

Run the script directly:

```bash
PYTHONPATH=./src python examples/benchmark_nm_vs_scipy.py
```

Configuration is intentionally simple and fixed in the file:

- `MAX_ITER`
- `MAX_EVALS`
- `STEP`
- `N_WORKERS`

## Output

The benchmark prints single-run metrics for each method:

- `best_f`,
- `gap`,
- `dist`,
- `time`.


The script includes:

- a simple convex 2D sphere objective,
- an explicit 7D shifted quadratic objective with:
  `loss = (x[0]-0.1)^2 + ... + (x[6]-0.7)^2`.
