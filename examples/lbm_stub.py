"""Visualise the LBM-stub objective.

Run from the repository root with:

    PYTHONPATH=./src python examples/lbm_stub.py

The plot illustrates the pseudo CFD response; it is **not physical**.
"""

import matplotlib.pyplot as plt
import numpy as np

from optilb.objectives import get_objective

f = get_objective("lbm_stub")

x = np.linspace(-1.0, 1.0, 50)
y = np.linspace(-1.0, 1.0, 50)
X, Y = np.meshgrid(x, y)
Z = np.empty_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        Z[i, j] = f(np.array([X[i, j], Y[i, j]]))

plt.figure(figsize=(5, 4))
plt.contourf(X, Y, Z, levels=20)
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("LBM stub objective")
plt.colorbar(label="cost")
plt.tight_layout()
plt.show()
