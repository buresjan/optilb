# optilb

**optilb** is a Pythonic optimisation toolbox for low-dimensional shape tuning.
It combines Latin-Hypercube sampling, pluggable local optimisers (Mesh Adaptive
Direct Search via PyNOMAD, SciPy's L-BFGS-B and a custom parallel
Nelder–Mead), and a catalogue of analytic benchmark objectives. Gaussian-Process
surrogates and robust optimisation utilities are tracked on the issue board, but the
current stack is already useful for plugging CFD/LBM objectives into a common
pipeline.

- Core dataclasses (`DesignSpace`, `DesignPoint`, `Constraint`, `OptResult`) keep
  optimisation metadata tidy and immutable.
- Optimisers share a lightweight `Optimizer` base that tracks evaluation
  history, budgets, and supports early stopping hooks.
- `OptimizationProblem` offers a façade that normalises configuration,
  forwards shared options (`max_evals`, `normalize`, `parallel`, `early_stopper`),
  and records an `OptimizationLog` for comparisons.
- `optilb.runner.run_with_schedule` executes an optimiser through successive
  scale levels to shrink steps or meshes in a controlled manner.

## Installation

`optilb` targets Python 3.10+. The short version is::

    git clone https://github.com/example/optilb.git
    cd optilb
    python -m venv .venv
    source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
    pip install .[examples]

Add the `nomad` extra to enable the NOMAD-based optimiser (requires the
`PyNomadBBO` package and its toolchain)::

    pip install .[nomad]

Developer tooling (formatters, type-checker, tests) lives behind the `dev`
extra::

    pip install .[dev]

See `INSTALL.md` for a complete walkthrough covering pip/venv and conda setups,
troubleshooting, and package extras.

To enable parallel evaluation of NOMAD trial points, construct the optimiser
with a desired worker count and set ``parallel=True`` when calling
``optimize``::

    from optilb.optimizers import MADSOptimizer
    opt = MADSOptimizer(n_workers=4)
    result = opt.optimize(obj, x0, ds, parallel=True)

`BFGSOptimizer` and `NelderMeadOptimizer` expose the same ``n_workers`` keyword
to bound the number of threads or processes used for parallel finite
difference/vertex evaluations.  All optimisers honour ``max_evals`` and accept
an optional ``EarlyStopper`` (with patience, target value, and time limit
controls) to terminate runs when progress stalls.  The ``runner.run_with_schedule``
helper executes an optimiser across multiple scale levels and clones any early
stopper so that each level tracks progress independently.

For a higher‑level entry point that makes cross‑method comparisons easy,
construct an :class:`OptimizationProblem`.  It accepts the objective, design
space and initial point along with a chosen optimiser and common controls such
as evaluation budgets and normalisation::

    import numpy as np
    
    from optilb import DesignSpace, OptimizationProblem, get_objective
    from optilb.optimizers import EarlyStopper

    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    prob = OptimizationProblem(
        obj,
        ds,
        np.array([3.0, 3.0]),
        optimizer="bfgs",
        max_evals=200,
        parallel=True,
        early_stopper=EarlyStopper(patience=8, eps=1e-4),
    )
    result = prob.run()
    print(result.best_x, result.best_f, prob.log.nfev, prob.log.early_stopped)

The toolbox ships core data classes, analytic objectives, a Latin-Hypercube
sampler and the optimisers mentioned above.  Below is a minimal sampling
example::

    from optilb import DesignSpace
    from optilb.sampling import lhs

    ds = DesignSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    points = lhs(4, ds, seed=123)
    for p in points:
        print(p.x)

``prob.log`` captures the optimiser class, runtime, evaluation count and whether
the run halted early (via ``EarlyStopper`` or the evaluation cap).  Planned
features and progress live in the GitHub issue tracker and project board.

## Examples

- `examples/basic_usage.py` – quick sanity checks.
- `examples/compare_optimisers.py` – benchmark built-in optimisers on toy objectives.
- `examples/plot_objectives.py` – visualise 2D objective functions.
- `examples/parallel_speedup.py` – demonstrate parallel Nelder–Mead speed-up on a slow objective.
