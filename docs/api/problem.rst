OptimizationProblem API
=======================

``OptimizationProblem`` orchestrates the optimisation run and exposes an
``OptimizationLog`` for post-analysis. When ``optimizer`` is a string alias
(``"bfgs"``, ``"nelder-mead"``, ``"mads"``) it constructs the corresponding
optimiser with ``optimizer_options``; otherwise the provided instance is used
directly. Common keyword arguments such as ``max_iter``, ``max_evals``,
``normalize``, ``parallel``, ``tol``, ``seed``, ``early_stopper`` and ``verbose``
are forwarded where supported. Additional solver-specific options can be
supplied through ``optimize_options``.

.. code-block:: python

    from optilb import DesignSpace, OptimizationProblem, get_objective
    from optilb.optimizers import EarlyStopper

    space = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    problem = OptimizationProblem(
        obj,
        space,
        [3.0, 3.0],
        optimizer="bfgs",
        max_evals=250,
        normalize=True,
        early_stopper=EarlyStopper(patience=8),
    )
    result = problem.run()
    print(problem.log.optimizer, problem.log.nfev, problem.log.early_stopped)

``problem.log`` is an ``OptimizationLog`` dataclass with the following fields:
``optimizer``, ``runtime`` (seconds), ``nfev`` and ``early_stopped``. When the
evaluation budget is reached or an ``EarlyStopper`` triggers, ``early_stopped``
becomes ``True``.
