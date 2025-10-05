Core Data Structures
====================

The :mod:`optilb.core` module defines lightweight dataclasses shared across
samplers, optimisers and façade helpers. They are immutable where practical so
optimisation history cannot be mutated accidentally.

DesignSpace
-----------

``DesignSpace`` stores lower and upper bounds (as read-only NumPy arrays) and an
optional tuple of variable names. Shapes must match and lower bounds must not
exceed the corresponding upper bounds.

.. code-block:: python

    from optilb import DesignSpace

    ds = DesignSpace(lower=[-1.0, -1.0], upper=[1.0, 1.0], names=("x", "y"))
    print(ds.dimension)  # 2

DesignPoint
-----------

``DesignPoint`` captures a single vector along with an optional tag and
timestamp. Points recorded by optimisers are stored as ``DesignPoint``
instances.

.. code-block:: python

    from optilb.core import DesignPoint

    p = DesignPoint([0.2, 0.4], tag="lhs")
    print(p.tag, p.timestamp.isoformat())

Constraint
----------

``Constraint`` wraps a callback accepting a NumPy array and returning either a
boolean (feasible/infeasible) or a floating penalty value.

.. code-block:: python

    from optilb.core import Constraint

    box = Constraint(lambda x: (x[0]**2 + x[1]**2) <= 1.0, name="unit_disc")

OptResult
---------

``OptResult`` is returned by every optimiser. It stores the best design vector,
its objective value, the full evaluation history, and the number of objective
evaluations (``nfev``). The ``best_x`` array is returned as a read-only view.

.. code-block:: python

    from optilb import OptResult

    res = OptResult(best_x=[0.0, 0.0], best_f=0.0)
    print(res.best_x.flags.writeable)  # False

OptimizationLog
---------------

``OptimizationProblem.run()`` stores summary metadata in an
``OptimizationLog`` with ``optimizer``, ``runtime``, ``nfev`` and
``early_stopped`` fields. The log is available via the ``log`` attribute after a
run.

Exceptions
----------

``optilb.exceptions.OptilbError`` is the base exception. Optimisers raise
``EvaluationBudgetExceeded`` when ``max_evals`` is hit; façade helpers catch this
condition and still return the best-known point.
