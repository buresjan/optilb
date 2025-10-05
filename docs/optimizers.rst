Local Optimizers
================

``optilb.optimizers`` bundles several local search algorithms that share a
common :class:`~optilb.optimizers.Optimizer` base class. Each optimiser records
every evaluated design point and returns an :class:`~optilb.OptResult` containing
the full history and evaluation count (``nfev``). A shared ``max_evals`` budget
is tracked inside the base class; when the budget is reached an
:class:`optilb.exceptions.EvaluationBudgetExceeded` exception is raised, the
best-known point is recorded, and façade helpers report the run as
early-stopped.

Common keyword arguments supported by most optimisers:

* ``max_iter`` – soft iteration limit (solver specific).
* ``max_evals`` – hard evaluation budget (enforced by the base class).
* ``parallel`` – enable multi-threaded or multi-process execution where supported.
* ``normalize`` – operate in the unit hypercube and map results back afterwards.
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

Built-in optimisers
-------------------

* :class:`optilb.optimizers.BFGSOptimizer` – wraps SciPy's L-BFGS-B. When
  ``normalize=True`` (default) it works in ``[0, 1]^d`` using a
  :class:`optilb.optimizers.utils.SpaceTransform`, records history in original
  coordinates, and supports numerical gradients via central differences. Use
  ``fd_eps`` (or the legacy alias ``step``) to set finite-difference steps; pass
  ``n_workers`` to parallelise gradient evaluations with threads when
  ``parallel=True``.
* :class:`optilb.optimizers.NelderMeadOptimizer` – derivative-free simplex search
  with optional normalisation and process-based parallelism. Objectives and
  constraints must be picklable when running with ``parallel=True``. The
  optimiser resamples the simplex after each iteration and honours constraint
  callbacks by applying the configured penalty.
* :class:`optilb.optimizers.MADSOptimizer` – interfaces with NOMAD's Mesh
  Adaptive Direct Search via the ``PyNomadBBO`` package. Pass ``normalize=True``
  to work in the unit cube (finite, non-degenerate bounds required). Provide
  ``n_workers`` to limit NOMAD's parallel evaluation threads.
* :class:`optilb.optimizers.EarlyStopper` – utility to halt optimisation when
  progress stalls. Reset it between runs (handled automatically by
  :class:`optilb.problem.OptimizationProblem` and
  :func:`optilb.runner.run_with_schedule`).

All optimisers expose the ``history`` property and ``budget_exhausted`` flag on
the base class. Use them to inspect the run after calling ``optimize``.
