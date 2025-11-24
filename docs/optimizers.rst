Local Optimizers
================

``optilb.optimizers`` bundles several local search algorithms that share a
common :class:`~optilb.optimizers.Optimizer` base class. Each optimiser records
every evaluated design point and returns an :class:`~optilb.OptResult`
containing the full history, a complete evaluation log, and the evaluation count
(``nfev``). A shared ``max_evals`` budget
is tracked inside the base class; when the budget is reached an
:class:`optilb.exceptions.EvaluationBudgetExceeded` exception is raised, the
best-known point is recorded, and façade helpers report the run as
early-stopped.

Common keyword arguments supported by most optimisers:

* ``max_iter`` – soft iteration limit (solver specific).
* ``max_evals`` – hard evaluation budget (enforced by the base class).
* ``parallel`` – enable multi-threaded or multi-process execution where supported.
* ``normalize`` – operate in the unit hypercube and map results back afterwards.
* ``memoize`` – cache completed evaluations to avoid recomputing the objective
  when an optimiser revisits the same point. Disabled by default and ignored for
  backends that cannot share cache state (such as PyNomad).
* ``early_stopper`` – an :class:`optilb.optimizers.EarlyStopper` instance with
  ``eps``, ``patience``, ``f_target``, ``time_limit`` and ``enabled`` controls.

.. code-block:: python

    from optilb import DesignSpace, get_objective
    from optilb.optimizers import BFGSOptimizer, EarlyStopper

    ds = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    opt = BFGSOptimizer(n_workers=2)
    res = opt.optimize(
        obj,
        ds.lower,
        ds,
        normalize=True,
        max_evals=150,
        early_stopper=EarlyStopper(patience=6, eps=1e-4),
        parallel=True,
    )
    print(res.best_x, res.best_f, res.nfev)
    print(len(res.evaluations))

Built-in optimisers
-------------------

* :class:`optilb.optimizers.BFGSOptimizer` – wraps SciPy's L-BFGS-B. When
  ``normalize=True`` (default) it works in ``[0, 1]^d`` using a
  :class:`optilb.optimizers.utils.SpaceTransform`, records history and evaluations in original
  coordinates, and supports numerical gradients via central differences. Use
  ``fd_eps`` (or the legacy alias ``step``) to set finite-difference steps; pass
  ``n_workers`` to parallelise gradient evaluations with threads when
  ``parallel=True``. Setting ``memoize=True`` reuses repeated evaluations during
  finite-difference sweeps.
* :class:`optilb.optimizers.NelderMeadOptimizer` – derivative-free simplex search
  with optional normalisation and process-based parallelism. Objectives and
  constraints must be picklable when running with ``parallel=True``. Set
  ``parallel_poll_points=True`` to speculatively evaluate reflection / expansion
  / contraction candidates each iteration (trading extra objective calls for
  lower latency). The optimiser resamples the simplex after each iteration,
  honours constraint callbacks by applying the configured penalty, and captures
  every evaluated simplex vertex even in process-based parallel runs.
  ``memoize=True`` enables a cache that short-circuits duplicate vertices across
  sequential, threaded, and process pools, including speculative batches run
  with ``parallel_poll_points=True``. Provide both ``initial_simplex`` and
  ``initial_simplex_values`` to start from a pre-evaluated simplex (dimension +
  1 vertices). Vertices are supplied in physical coordinates and are mapped to
  the unit cube automatically when ``normalize=True``. The initial simplex
  evaluation phase is skipped in that case; bounds and constraints are checked
  and infeasible vertices receive the configured penalty.
* :class:`optilb.optimizers.MADSOptimizer` – interfaces with NOMAD's Mesh
  Adaptive Direct Search via the ``PyNomadBBO`` package. Pass ``normalize=True``
  to work in the unit cube (finite, non-degenerate bounds required). Provide
  ``n_workers`` to limit NOMAD's parallel evaluation threads. All evaluations
  reported by NOMAD are stored in original coordinates for post-analysis.
  Memoisation is currently ignored because evaluations are handled entirely by
  PyNomad. Normalisation improves conditioning on highly anisotropic problems,
  but PyNomad's mesh schedule can require larger evaluation budgets to reach the
  same optimum quality as an unscaled run; raise ``max_iter``/``max_evals`` when
  needed.
* :class:`optilb.optimizers.EarlyStopper` – utility to halt optimisation when
  progress stalls. Reset it between runs (handled automatically by
  :class:`optilb.problem.OptimizationProblem` and
  :func:`optilb.runner.run_with_schedule`).

All optimisers expose ``history``, ``evaluations`` and the ``budget_exhausted``
flag on the base class. Use them to inspect the run after calling ``optimize``.

Nelder–Mead with a predefined simplex
-------------------------------------

.. code-block:: python

   import numpy as np
   from optilb import DesignSpace, get_objective
   from optilb.optimizers import NelderMeadOptimizer

   space = DesignSpace(lower=np.array([0.0, -5.0]), upper=np.array([10.0, 5.0]))
   objective = get_objective("quadratic")

   simplex = [
       np.array([0.0, -5.0]),
       np.array([2.0, -3.0]),
       np.array([4.0, -1.0]),
   ]
   simplex_values = [objective(v) for v in simplex]

   opt = NelderMeadOptimizer()
   res = opt.optimize(
       objective,
       x0=simplex[0],
       space=space,
       initial_simplex=simplex,
       initial_simplex_values=simplex_values,
       normalize=True,  # vertices are mapped to the unit cube automatically
       max_iter=50,
   )
   print(res.best_x, res.best_f, res.nfev)
