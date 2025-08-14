OptimizationProblem API
=======================

``OptimizationProblem`` orchestrates the optimisation run and exposes a small
log structure for post‑analysis.

.. code-block:: python

    from optilb import OptimizationProblem, DesignSpace, get_objective

    space = DesignSpace(lower=[-5.0, -5.0], upper=[5.0, 5.0])
    obj = get_objective("quadratic")
    problem = OptimizationProblem(obj, space, [3.0, 3.0], optimizer="bfgs")
    result = problem.run()
    print(problem.log.optimizer, problem.log.nfev)

The ``run`` method returns an :class:`OptResult` instance capturing the best
point and evaluation history.
