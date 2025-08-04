# optilb

**optilb** is an optimisation toolbox for low-dimensional shape tuning.  The
roadmap combines Latin-Hypercube sampling with pluggable optimisers (Mesh
Adaptive Direct Search via PyNOMAD, SciPy's L-BFGS-B and a custom parallel
Nelder–Mead).  CFD objectives—from analytic toy cases to external executables—are
wrapped through a unified `OptimizationProblem` interface.  Optional
Gaussian‑Process surrogates and robust optimisation utilities are scheduled for
later milestones.

## Installation

`optilb` targets Python 3.10+ and can be installed from source::

    git clone https://github.com/example/optilb.git
    cd optilb
    pip install -r requirements.txt

The `PyNomadBBO` package is optional but required for the `MADSOptimizer`::

    pip install PyNomadBBO

To enable parallel evaluation of NOMAD trial points, construct the optimiser
with a desired worker count and set ``parallel=True`` when calling
``optimize``::

    from optilb.optimizers import MADSOptimizer
    opt = MADSOptimizer(n_workers=4)
    result = opt.optimize(obj, x0, ds, parallel=True)

`BFGSOptimizer` and `NelderMeadOptimizer` expose the same ``n_workers`` keyword
to bound the number of threads or processes used for parallel finite
difference/vertex evaluations.

The current codebase provides the core data classes and a Latin-Hypercube
sampler.  Below is a minimal example::

    from optilb import DesignSpace
    from optilb.sampling import lhs

    ds = DesignSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    points = lhs(4, ds, seed=123)
    for p in points:
        print(p.x)

See `ROADMAP.md` for planned features and progress.

## Examples

- `examples/basic_usage.py` – quick sanity checks.
- `examples/compare_optimisers.py` – benchmark built-in optimisers on toy objectives.
- `examples/plot_objectives.py` – visualise 2D objective functions.
- `examples/parallel_speedup.py` – demonstrate parallel Nelder–Mead speed-up on a slow objective.
