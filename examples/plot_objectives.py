"""Plot 2D objective functions for visual inspection.

Run from the repository root with::

    PYTHONPATH=./src python examples/plot_objectives.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from optilb import get_objective

OPTIMUM_VALUES = {
    "quadratic": 0.0,
    "rastrigin": 0.0,
    "checkerboard": -1.0,
    "step_rastrigin": 0.0,
    "lbm_stub": -0.1,
    "noisy_discontinuous": float("-inf"),
}

OBJECTIVES = [
    "quadratic",
    "rastrigin",
    "noisy_discontinuous",
    "checkerboard",
    "step_rastrigin",
    "lbm_stub",
]

for name in OBJECTIVES:
    f = get_objective(name)
    x = np.linspace(-2.0, 2.0, 50)
    y = np.linspace(-2.0, 2.0, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.empty_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f(np.array([X[i, j], Y[i, j]]))
    plt.figure(figsize=(5, 4))
    plt.contourf(X, Y, Z, levels=20)
    plt.xlabel("x1")
    plt.ylabel("x2")
    optimum = OPTIMUM_VALUES.get(name, float("nan"))
    if np.isfinite(optimum):
        title = f"{name} objective (min {optimum:.2f})"
    else:
        title = f"{name} objective (min unbounded)"
    plt.title(title)
    plt.colorbar(label="cost")
    plt.tight_layout()
    plt.show()
