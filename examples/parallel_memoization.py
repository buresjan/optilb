from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np


def _ensure_repo_import() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_repo_import()

from optilb import DesignSpace  # noqa: E402
from optilb.optimizers import NelderMeadOptimizer  # noqa: E402


def slow_quadratic(x: np.ndarray) -> float:
    """Deliberately slow objective to make memoization impact visible."""
    time.sleep(0.05)
    return float(np.sum(x**2))


def run_trial(*, memoize: bool) -> tuple[float, int]:
    """Execute a tiny optimisation run and report elapsed time and nfev."""
    optimizer = NelderMeadOptimizer(step=0.0, memoize=memoize, n_workers=2)
    space = DesignSpace(
        lower=np.array([-1.0, -1.0], dtype=float),
        upper=np.array([1.0, 1.0], dtype=float),
    )
    start = time.perf_counter()
    result = optimizer.optimize(
        slow_quadratic,
        x0=np.array([0.2, -0.4], dtype=float),
        space=space,
        max_iter=0,
        parallel=True,
        normalize=False,
    )
    elapsed = time.perf_counter() - start
    return elapsed, result.nfev


def main() -> None:
    no_cache_time, no_cache_evals = run_trial(memoize=False)
    cache_time, cache_evals = run_trial(memoize=True)

    print("Parallel Nelder-Mead with memoization toggle")
    print("------------------------------------------------")
    print(f"memoize=False -> {no_cache_evals} evaluations in {no_cache_time:.3f}s")
    print(f"memoize=True  -> {cache_evals} evaluations in {cache_time:.3f}s")
    if cache_time > 0:
        speedup = no_cache_time / cache_time
        eval_factor = (
            float(no_cache_evals) / cache_evals if cache_evals > 0 else float("inf")
        )
        print(f"Runtime speed-up: {speedup:.2f}x")
        print(f"Evaluation reduction: {eval_factor:.2f}x fewer objective calls")
    else:
        print("Cache-enabled run finished too quickly to measure a speed-up.")


if __name__ == "__main__":
    main()
