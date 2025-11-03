from __future__ import annotations

import os
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

os.environ.setdefault("OPTILB_FORCE_THREAD_POOL", "1")

from optilb import DesignSpace  # noqa: E402
from optilb.optimizers import NelderMeadOptimizer  # noqa: E402


def run_experiment(memoize: bool) -> tuple[float, int, int]:
    calls = {"value": 0}

    def expensive_objective(x: np.ndarray) -> float:
        calls["value"] += 1
        time.sleep(0.05)
        return float(np.sum(x**2))

    optimizer = NelderMeadOptimizer(step=0.0, memoize=memoize, n_workers=1)
    space = DesignSpace(lower=np.array([-2.0, -2.0]), upper=np.array([2.0, 2.0]))
    start = time.perf_counter()
    result = optimizer.optimize(
        expensive_objective,
        x0=np.array([0.5, -0.5]),
        space=space,
        max_iter=2,
        normalize=False,
        parallel=True,
    )
    elapsed = time.perf_counter() - start
    return elapsed, calls["value"], result.nfev


def main() -> None:
    no_cache_time, no_cache_calls, no_cache_nfev = run_experiment(memoize=False)
    cache_time, cache_calls, cache_nfev = run_experiment(memoize=True)

    speedup = no_cache_time / cache_time if cache_time > 0 else float("inf")
    print("Memoization disabled:")
    print(f"  Runtime: {no_cache_time:.3f}s, objective calls: {no_cache_calls}, nfev: {no_cache_nfev}")
    print("Memoization enabled:")
    print(f"  Runtime: {cache_time:.3f}s, objective calls: {cache_calls}, nfev: {cache_nfev}")
    print(f"Speed-up factor: {speedup:.2f}x")


if __name__ == "__main__":
    main()
