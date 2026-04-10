from __future__ import annotations

import time

import numpy as np
from scipy import optimize

from optilb import DesignSpace
from optilb.optimizers import NelderMeadOptimizer


# Fixed benchmark settings (edit here if you want a different setup)
MAX_ITER = 300
MAX_EVALS = 500
STEP = 0.2
N_WORKERS = 6


def sphere(x: np.ndarray) -> float:
    """Simple convex benchmark."""

    arr = np.asarray(x, dtype=float)
    return float(np.sum(arr**2))


def shifted_seven_dimensional(x: np.ndarray) -> float:
    """Explicit 7D quadratic target requested by the user."""

    x = np.asarray(x, dtype=float)
    loss = (
        (x[0] - 0.1) ** 2
        + (x[1] - 0.2) ** 2
        + (x[2] - 0.3) ** 2
        + (x[3] - 0.4) ** 2
        + (x[4] - 0.5) ** 2
        + (x[5] - 0.6) ** 2
        + (x[6] - 0.7) ** 2
    )
    return float(loss)


def build_initial_simplex(x0: np.ndarray, step: float) -> np.ndarray:
    """Use the same simplex shape in SciPy and optilb."""

    dim = x0.size
    simplex = np.empty((dim + 1, dim), dtype=float)
    simplex[0] = x0
    for i in range(dim):
        point = x0.copy()
        point[i] += step
        simplex[i + 1] = point
    return simplex


def run_scipy_nm(
    objective,
    x0: np.ndarray,
    simplex: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    t0 = time.perf_counter()
    result = optimize.minimize(
        objective,
        x0,
        method="Nelder-Mead",
        options={
            "adaptive": False,
            "initial_simplex": simplex,
            "maxiter": MAX_ITER,
            "maxfev": MAX_EVALS,
            "xatol": 0.0,
            "fatol": 0.0,
            "disp": False,
        },
    )
    elapsed = time.perf_counter() - t0
    return (
        np.asarray(result.x, dtype=float),
        float(result.fun),
        elapsed,
    )


def run_optilb_nm(
    objective,
    x0: np.ndarray,
    space: DesignSpace,
    *,
    parallel: bool,
) -> tuple[np.ndarray, float, float]:
    optimizer = NelderMeadOptimizer(
        step=STEP,
        alpha=1.0,
        gamma=2.0,
        beta=0.5,
        delta=0.5,
        sigma=0.5,
        no_improve_thr=0.0,
        no_improv_break=MAX_ITER + 1,
        n_workers=N_WORKERS,
        parallel_poll_points=False,
        memoize=False,
    )

    t0 = time.perf_counter()
    result = optimizer.optimize(
        objective,
        x0,
        space,
        max_iter=MAX_ITER,
        max_evals=MAX_EVALS,
        tol=0.0,
        parallel=parallel,
        normalize=False,
    )
    elapsed = time.perf_counter() - t0
    return (
        np.asarray(result.best_x, dtype=float),
        float(result.best_f),
        elapsed,
    )


def run_problem(
    name: str,
    objective,
    x0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    optimum_x: np.ndarray,
) -> None:
    x0 = np.asarray(x0, dtype=float)
    space = DesignSpace(lower=lower, upper=upper)
    optimum_f = float(objective(optimum_x))
    simplex = build_initial_simplex(x0, STEP)

    scipy_x, scipy_f, scipy_time = run_scipy_nm(objective, x0, simplex)
    serial_x, serial_f, serial_time = run_optilb_nm(
        objective,
        x0,
        space,
        parallel=False,
    )
    parallel_x, parallel_f, parallel_time = run_optilb_nm(
        objective,
        x0,
        space,
        parallel=True,
    )

    rows = [
        ("SciPy-NM", scipy_x, scipy_f, scipy_time),
        ("optilb-NM (serial)", serial_x, serial_f, serial_time),
        ("optilb-NM (parallel)", parallel_x, parallel_f, parallel_time),
    ]

    print(f"\n{name}")
    print(f"  x0={x0}")
    for method, best_x, best_f, runtime in rows:
        gap = float(best_f - optimum_f)
        dist = float(np.linalg.norm(best_x - optimum_x))
        print(
            f"  {method:20s} best_f={best_f:.3e} "
            f"gap={gap:.3e} dist={dist:.3e} time={runtime:.3e}s"
        )


def main() -> None:
    print("Fairness controls:")
    print("- same objective and same start points")
    print("- same simplex geometry")
    print("- same Nelder-Mead coefficients")
    print("- same max_iter and max_evals")
    print("- method-specific extras disabled")

    run_problem(
        name="Sphere (2D)",
        objective=sphere,
        x0=np.array([2.0, -1.5]),
        lower=-5.0 * np.ones(2),
        upper=5.0 * np.ones(2),
        optimum_x=np.zeros(2),
    )

    run_problem(
        name="Shifted quadratic (7D explicit loss)",
        objective=shifted_seven_dimensional,
        x0=np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]),
        lower=-5.0 * np.ones(7),
        upper=5.0 * np.ones(7),
        optimum_x=np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]),
    )


if __name__ == "__main__":
    main()
