"""Small ``OptimizationProblem`` walkthrough.

Run this module from the repository root with:

    PYTHONPATH=./src python examples/demo.py

It compares a few built-in optimizers on the same 7D quadratic target. The
optional MADS run is skipped automatically when ``PyNomadBBO`` is not
installed.
"""

from __future__ import annotations

import numpy as np

from optilb import DesignSpace, OptimizationProblem
from optilb.optimizers import MADSOptimizer


def shifted_quadratic(x: np.ndarray) -> float:
    """Seven-dimensional quadratic with a non-zero minimizer."""

    target = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], dtype=float)
    arr = np.asarray(x, dtype=float)
    return float(np.sum((arr - target) ** 2))


def main() -> None:
    x0 = np.full(7, 0.5, dtype=float)
    space = DesignSpace(lower=np.zeros(7), upper=np.ones(7))

    cases: list[tuple[str, dict[str, object]]] = [
        (
            "BFGS",
            {
                "optimizer": "bfgs",
                "parallel": True,
                "normalize": False,
            },
        ),
        (
            "Nelder-Mead",
            {
                "optimizer": "nm",
                "parallel": False,
                "normalize": False,
                "optimizer_options": {"no_improv_break": 200},
            },
        ),
    ]

    if MADSOptimizer.is_available():
        cases.append(
            (
                "MADS",
                {
                    "optimizer": "mads",
                    "parallel": True,
                    "normalize": False,
                },
            )
        )

    print("Comparing OptimizationProblem across available optimizers")
    print(f"start_x={x0}")
    for label, kwargs in cases:
        problem = OptimizationProblem(
            shifted_quadratic,
            space,
            x0,
            max_iter=1000,
            max_evals=1000,
            tol=1e-12,
            **kwargs,
        )
        result = problem.run()
        print(
            f"{label:12s} best_f={result.best_f:.3e} "
            f"best_x={np.array2string(result.best_x, precision=3)}"
        )

    if not MADSOptimizer.is_available():
        print("MADS skipped: optional dependency PyNomadBBO is not installed.")


if __name__ == "__main__":
    main()
