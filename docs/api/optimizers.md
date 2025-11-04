# Optimizers API

`optilb.optimizers` defines the common `Optimizer` base class that all local
search methods implement. It tracks an evaluation counter, optional budgets
(`max_evals`), and records both a tuple of `DesignPoint` instances (via
`history`) and a full evaluation log (via `evaluations`, holding
`EvaluationRecord` objects). When enabled with `memoize=True`, the base class can
cache objective values for repeated points (where supported). When a budget is
exhausted an `EvaluationBudgetExceeded` exception is raised; façade helpers catch
it and surface the best-known point while marking the run as early stopped.

`EarlyStopper` provides patience-, target-, and time-based stopping criteria and
is safe to reuse once `reset()` has been called (handled automatically by
`OptimizationProblem` and `run_with_schedule`).

```python
from optilb.optimizers import BFGSOptimizer, EarlyStopper

stopper = EarlyStopper(patience=5, eps=1e-4, f_target=0.0)
result = BFGSOptimizer().optimize(
    obj,
    x0,
    ds,
    normalize=True,
    max_evals=200,
    early_stopper=stopper,
    parallel=True,
)
print(result.best_x, result.best_f, result.nfev)
```

## Built-in implementations

- `BFGSOptimizer` – wraps SciPy's L-BFGS-B algorithm. Numerical gradients default
  to central differences with step size `fd_eps` (or legacy alias `step`);
  setting `n_workers` allows threaded gradient evaluations when `parallel=True`.
  With `normalize=True` the design space is mapped to `[0, 1]^d` before calling
  SciPy and history/evaluations are mapped back afterwards. Enable memoisation
  via `memoize=True` to reuse duplicate evaluations encountered during
  finite-difference passes.
- `NelderMeadOptimizer` – derivative-free simplex search supporting optional
  normalisation and multi-process evaluation of simplex vertices. Objectives and
  constraints must be picklable when `parallel=True`. Set
  `parallel_poll_points=True` to speculatively score reflection / expansion /
  contraction candidates together when running in parallel. All evaluated simplex
  vertices are captured in `evaluations`, even when running across processes.
  With `memoize=True`, duplicate vertices are short-circuited across sequential,
  threaded, and process-based executions, including speculative batches emitted
  by `parallel_poll_points=True`.
- `MADSOptimizer` – interfaces with NOMAD's Mesh Adaptive Direct Search via the
  external `PyNomadBBO` package. Use `n_workers` to control NOMAD's evaluation
  threads and `normalize=True` to operate in the unit cube (finite bounds
  required). Reported evaluations are converted back to the original space
  before being logged. Memoisation requests are ignored because evaluations are
  delegated to PyNomad. Normalising highly anisotropic problems can improve
  conditioning, but the mesh schedule may need more evaluations (`max_iter` /
  `max_evals`) to recover the same solution quality as an unscaled run.

All optimisers accept a `parallel` flag, `max_iter`, `max_evals`, `tol`, `seed`
(where applicable), and optional constraint callbacks. History, `evaluations`,
and the `budget_exhausted` flag can be inspected after a run to understand
whether the solver terminated naturally or due to limits.
