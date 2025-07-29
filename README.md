# optilb

**optilb** is an optimisation toolbox for low-dimensional shape tuning.  The
roadmap combines Latin-Hypercube sampling with pluggable optimisers (Mesh
Adaptive Direct Search via PyNOMAD, SciPy's L-BFGS-B and a custom parallel
Nelder–Mead).  CFD objectives—from analytic toy cases to external executables—are
wrapped through a unified `OptimizationProblem` interface.  Optional
Gaussian‑Process surrogates and robust optimisation utilities are scheduled for
later milestones.

To use the MADS optimiser you need the external `PyNomadBBO` package::

    pip install PyNomadBBO

The current codebase provides the core data classes and a Latin-Hypercube
sampler.  Below is a minimal example::

    from optilb import DesignSpace
    from optilb.sampling import lhs

    ds = DesignSpace(lower=[0.0, 0.0], upper=[1.0, 1.0])
    points = lhs(4, ds, seed=123)
    for p in points:
        print(p.x)

See `ROADMAP.md` for planned features and progress.
