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
``lbm_stub``. The last one is a purely numerical surrogate for an LBM solver and
**is not physically accurate**.

Factory helpers (``make_noisy_discontinuous``, ``make_spiky_sine``,
``make_checkerboard``, ``make_step_rastrigin``) accept ``sigma`` and ``seed``
parameters so you can control stochastic behaviour::

    from optilb.objectives import make_spiky_sine

    obj = make_spiky_sine(sigma=0.0)

Use explicit factory functions when you need several independent noisy
instances; ``get_objective("spiky_sine", seed=123)`` simply forwards keyword
arguments to the underlying factory.
