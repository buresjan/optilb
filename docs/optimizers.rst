Local Optimizers
===============

``optilb`` bundles several local search algorithms through the :mod:`optilb.optimizers` package. Each optimizer implements the :class:`optilb.optimizers.Optimizer` base interface and returns an :class:`optilb.OptResult`.

Example usage::

    from optilb import DesignSpace, get_objective
    from optilb.optimizers import BFGSOptimizer

    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = BFGSOptimizer()
    res = opt.optimize(obj, ds.lower, ds, normalize=True)
    print(res.best_x, res.best_f)

Available optimizers:

* :class:`optilb.optimizers.BFGSOptimizer` – wraps SciPy's L-BFGS-B for smooth objectives and can normalise the design space with ``normalize=True``.
* :class:`optilb.optimizers.NelderMeadOptimizer` – supports optional parallel evaluation and normalisation.
* :class:`optilb.optimizers.MADSOptimizer` – interfaces with NOMAD's Mesh Adaptive Direct Search (requires ``PyNomadBBO``) and can normalise the search space with ``normalize=True`` (requires finite, non-degenerate bounds and reports results in the original coordinates).
* :class:`optilb.optimizers.EarlyStopper` – utility to halt optimisation when progress stalls.

All optimizers that support parallel execution accept an ``n_workers`` keyword to limit resource usage.
