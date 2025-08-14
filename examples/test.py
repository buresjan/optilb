from __future__ import annotations

import time
from functools import partial
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np

from optilb import DesignSpace, OptimizationProblem, OptResult
from optilb.optimizers.mads import PyNomad

# Global configuration shared by all optimisation runs
SCALE = 0.15
MAX_ITER = 200
N_WORKERS = 4
FD_EPS = 1e-6
SLEEP_TIME = 0.1


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------


def rugged_objective(
    x: np.ndarray, *, continuous: bool = True, scale: float = SCALE
) -> float:
    """Rugged multimodal function with smooth or quantized output."""

    time.sleep(SLEEP_TIME)

    # Base continuous function: mix of sin waves and a parabola (multimodal, smooth)
    x1, x2 = x
    smooth_val = np.sin(3 * x1) * np.sin(3 * x2) + 0.1 * (x1**2 + x2**2)
    if continuous:
        return smooth_val
    # Quantize x1, x2 to nearest grid centre of size ``scale``
    x1q = (np.floor(x1 / scale) + 0.5) * scale
    x2q = (np.floor(x2 / scale) + 0.5) * scale
    return np.sin(3 * x1q) * np.sin(3 * x2q) + 0.1 * (x1q**2 + x2q**2)


# ---------------------------------------------------------------------------
# Visualise both continuous and quantised versions
# ---------------------------------------------------------------------------

x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)

Z_continuous = np.vectorize(
    lambda a, b: rugged_objective((a, b), continuous=True)  # type: ignore[arg-type]
)(X, Y)
Z_constant = np.vectorize(
    lambda a, b: rugged_objective((a, b), continuous=False)  # type: ignore[arg-type]
)(X, Y)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(
    Z_continuous,
    extent=[x.min(), x.max(), y.min(), y.max()],
    origin="lower",
    cmap="viridis",
)
axes[0].set_title("Continuous version")
fig.colorbar(im1, ax=axes[0])

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


# ---------------------------------------------------------------------------
# Optimisation set-up
# ---------------------------------------------------------------------------

space = DesignSpace(lower=[-1.5, -1.5], upper=[1.5, 1.5])

# Initial points to probe
initial_points = [
    np.array([0.001, -0.001]),
    np.array([1.0, -1.0]),
    np.array([-1.0, 1.0]),
]

# Wrap objectives so they are picklable in worker processes
obj_cont: Callable[[np.ndarray], float] = rugged_objective
obj_disc: Callable[[np.ndarray], float] = partial(
    rugged_objective, continuous=False, scale=SCALE
)

# Determine MADS availability
HAS_MADS = PyNomad is not None  # pragma: no cover - optional dependency

# Store optimisation results and wall times for later comparison
results: dict[str, list[tuple[str, OptResult, float]]] = {}


def run(
    name: str,
    optimizer: str,
    objective: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    opt_opts: dict | None = None,
    key: str,
) -> OptResult:
    opts = dict(opt_opts or {})
    prob = OptimizationProblem(
        objective,
        space,
        x0,
        optimizer=optimizer,
        max_iter=MAX_ITER,
        tol=1e-6,
        parallel=True,
        normalize=True,
        optimizer_options=opts,
    )
    start = time.perf_counter()
    res = prob.run()
    elapsed = time.perf_counter() - start
    results.setdefault(key, []).append((name, res, elapsed))
    print(
        f"{name}: f_min = {res.best_f:.4f} at x = {res.best_x}, "
        f"evals = {res.nfev},  time = {elapsed:.2f}s",
    )
    return res


# ---------------------------------------------------------------------------
# Run optimisation from multiple initial points
# ---------------------------------------------------------------------------

for x0 in initial_points:
    key = np.array2string(x0, precision=3)
    print(f"\n=== Starting from {key} ===")
    run(
        "BFGS on continuous",
        "bfgs",
        obj_cont,
        x0,
        opt_opts={"n_workers": N_WORKERS, "fd_eps": FD_EPS},
        key=key,
    )
    run(
        "BFGS on piecewise-constant",
        "bfgs",
        obj_disc,
        x0,
        opt_opts={"n_workers": N_WORKERS, "fd_eps": FD_EPS},
        key=key,
    )
    if HAS_MADS:
        run(
            "MADS on piecewise-constant",
            "mads",
            obj_disc,
            x0,
            opt_opts={"n_workers": N_WORKERS},
            key=key,
        )
    else:
        print("MADS optimizer not available (PyNomadBBO not installed).")
    run(
        "Nelder-Mead on piecewise-constant",
        "nelder-mead",
        obj_disc,
        x0,
        opt_opts={"n_workers": N_WORKERS},
        key=key,
    )


print("\nSummary:")
for key, runs in results.items():
    print(f"\nInitial point {key}:")
    for name, res, elapsed in runs:
        print(
            f"  {name}: f_min = {res.best_f:.4f} at x = {res.best_x}, "
            f"evals = {res.nfev},  time = {elapsed:.2f}s",
        )
