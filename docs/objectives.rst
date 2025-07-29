Analytic Objective Functions
===========================

``optilb`` ships several toy objective functions useful for quick benchmarking.
They are retrieved via :func:`optilb.get_objective`.

Example::

    from optilb import get_objective
    import numpy as np

    quad = get_objective("quadratic")
    print(quad(np.array([0.0, 0.0])))  # 0.0

Available names are ``quadratic``, ``rastrigin``, ``noisy_discontinuous`` and
``plateau_cliff``. See the docstrings for details.
