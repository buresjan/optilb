Analytic Objective Functions
===========================

``optilb`` ships several toy objective functions useful for quick benchmarking.
They are retrieved via :func:`optilb.get_objective`.

Example::

    from optilb import get_objective
    import numpy as np

    quad = get_objective("quadratic")
    print(quad(np.array([0.0, 0.0])))  # 0.0

Available names are ``quadratic``, ``rastrigin``, ``noisy_discontinuous``,
``checkerboard``, ``step_rastrigin``, ``spiky_sine``, ``plateau_cliff`` and
``lbm_stub``. The last one is a purely numerical
surrogate for an LBM solver and **is not physically accurate**.
See the docstrings for details.
