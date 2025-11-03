from __future__ import annotations

from pathlib import Path
import sys

import numpy as np


def _ensure_repo_import() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


_ensure_repo_import()

from optilb import DesignSpace, get_objective  # noqa: E402
from optilb.optimizers import MADSOptimizer, NelderMeadOptimizer  # noqa: E402
from optilb.optimizers.mads import PYNOMAD_AVAILABLE  # noqa: E402


def _write_log(result, path: Path) -> None:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# idx\t" + "\t".join(f"x{i+1}" for i in range(result.best_x.size)))
        handle.write("\tf(x)\n")
        for idx, record in enumerate(result.evaluations):
            coords = "\t".join(f"{value:.6f}" for value in record.x)
            handle.write(f"{idx:04d}\t{coords}\t{record.value:.6f}\n")


def _run_nelder_mead(ds: DesignSpace, objective) -> None:
    optimizer = NelderMeadOptimizer()
    result = optimizer.optimize(
        objective=objective,
        x0=np.array([1.8, -1.8]),
        space=ds,
        max_iter=60,
        normalize=True,
        parallel=False,
    )
    log_path = Path(__file__).with_name("evaluation_log_nm.txt")
    _write_log(result, log_path)
    print(f"Wrote {len(result.evaluations)} evaluations to {log_path}")


def _run_mads(ds: DesignSpace, objective) -> None:
    if not PYNOMAD_AVAILABLE:
        print("PyNomad not available; skipping MADS evaluation log.")
        return

    optimizer = MADSOptimizer()
    result = optimizer.optimize(
        objective=objective,
        x0=np.array([1.0, 1.5]),
        space=ds,
        max_iter=80,
        normalize=True,
        parallel=False,
    )
    log_path = Path(__file__).with_name("evaluation_log_mads.txt")
    _write_log(result, log_path)
    print(f"Wrote {len(result.evaluations)} evaluations to {log_path}")


def main() -> None:
    ds = DesignSpace(lower=np.array([-2.0, -2.0]), upper=np.array([2.0, 2.0]))
    objective = get_objective("quadratic")
    _run_nelder_mead(ds, objective)
    _run_mads(ds, objective)


if __name__ == "__main__":
    main()
