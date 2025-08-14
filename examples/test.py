from __future__ import annotations

import time
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from optilb import DesignSpace, OptResult
from optilb.optimizers import BFGSOptimizer, MADSOptimizer, NelderMeadOptimizer

# Global configuration shared by all optimisation runs
SCALE = 0.15
MAX_ITER = 200
N_WORKERS = 4
FD_EPS = 1e-6
SLEEP_TIME = 0.1


# Define a 2D rugged test function (continuous and piecewise-constant versions)
def rugged_objective(
    x: np.ndarray, continuous: bool = True, scale: float = SCALE
) -> float:
    """Rugged multimodal function with smooth or quantized output.

    Parameters
    ----------
    x:
        2D design point.
    continuous:
        Whether to evaluate the smooth version of the objective.
    scale:
        Cell size for the piecewise-constant variant.
    """
    time.sleep(SLEEP_TIME)

    # Base continuous function: mix of sin waves and a parabola (multimodal, smooth)
    x1, x2 = x  # two parameters
    smooth_val = np.sin(3 * x1) * np.sin(3 * x2) + 0.1 * (x1**2 + x2**2)
    if continuous:
        return smooth_val
    # Quantize x1, x2 to nearest grid center of size `scale`
    x1q = (np.floor(x1 / scale) + 0.5) * scale
    x2q = (np.floor(x2 / scale) + 0.5) * scale
    return np.sin(3 * x1q) * np.sin(3 * x2q) + 0.1 * (x1q**2 + x2q**2)


# Create a grid
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)

# Evaluate both versions
Z_continuous = np.vectorize(
    lambda a, b: rugged_objective((a, b), continuous=True)  # type: ignore[arg-type]
)(X, Y)
Z_constant = np.vectorize(
    lambda a, b: rugged_objective((a, b), continuous=False)  # type: ignore[arg-type]
)(X, Y)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Continuous
im1 = axes[0].imshow(
    Z_continuous,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="viridis",
)
axes[0].set_title("Continuous version")
fig.colorbar(im1, ax=axes[0])

# Piecewise-constant
im2 = axes[1].imshow(
    Z_constant,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="viridis",
)
axes[1].set_title("Piecewise-constant version")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# Set up a 2D design space for the parameters (e.g., both in [-1.5, 1.5])
space = DesignSpace(lower=[-1.5, -1.5], upper=[1.5, 1.5])

# Initial guess for optimization
x0 = np.array([0.001, -0.001])

# Create optimizer instances
opt_bfgs = BFGSOptimizer(n_workers=N_WORKERS, fd_eps=FD_EPS)
opt_nm = NelderMeadOptimizer(n_workers=N_WORKERS)
try:
    opt_mads = MADSOptimizer(n_workers=N_WORKERS)
except ImportError:
    opt_mads = None

# Wrap objectives with ``functools.partial`` so they are picklable when
# evaluated in parallel worker processes.
obj_cont = rugged_objective
obj_disc = partial(rugged_objective, continuous=False, scale=SCALE)

# Store optimisation results and wall times for later comparison
results: list[tuple[str, OptResult, float]] = []


def run(name: str, optimizer, objective):
    start = time.perf_counter()
    res = optimizer.optimize(
        objective,
        x0,
        space,
        max_iter=MAX_ITER,
        tol=1e-6,
        parallel=True,
    )
    elapsed = time.perf_counter() - start
    results.append((name, res, elapsed))
    print(
        f"{name}: f_min = {res.best_f:.4f} at x = {res.best_x}, "
        f"evals = {res.nfev},  time = {elapsed:.2f}s"
    )
    return res


# Run BFGS on the continuous (smooth) objective
run("BFGS on continuous", opt_bfgs, obj_cont)

# Run BFGS on the piecewise-constant objective (expected to struggle)
run("BFGS on piecewise-constant", opt_bfgs, obj_disc)


# Run MADS on the piecewise-constant objective (if available)
if opt_mads:
    run("MADS on piecewise-constant", opt_mads, obj_disc)
else:
    print("MADS optimizer not available (PyNomadBBO not installed).")


# Run Nelder–Mead on the piecewise-constant objective
run("Nelder-Mead on piecewise-constant", opt_nm, obj_disc)

print("\nSummary:")
for name, res, elapsed in results:
    print(
        f"{name}: f_min = {res.best_f:.4f} at x = {res.best_x}, "
        f"evals = {res.nfev},  time = {elapsed:.2f}s"
    )
