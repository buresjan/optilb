# Installation Guide

This document describes two supported workflows for installing **optilb** on a
local machine:

- A reproducible Conda environment that captures Python and binary
  dependencies.
- A lightweight virtual environment using the Python standard library.

Both paths install the package in editable mode so source edits take effect
immediately--ideal for development and optimisation experiments.

---

## 1. Prerequisites

- **Operating system**: Linux, macOS, or Windows.
- **Python**: Version 3.10, 3.11, or 3.12.
- **Build tools**: Only required if you opt into the NOMAD optimiser (`pip install '.[nomad]'`). PyNomadBBO ships wheels for common platforms, but a C/C++ toolchain may be needed elsewhere.
- **Git**: Optional but recommended for cloning the repository.

If you plan to use the Conda workflow, install either Anaconda, Miniconda, or
Mambaforge. For the native virtual environment path, ensure `python` and `pip`
are on your `PATH`.

---

## 2. Clone the repository

```bash
git clone https://github.com/example/optilb.git
cd optilb
```

If you already have a local checkout, update it instead:

```bash
git pull
```

---

## 3. Option A - Conda environment (recommended for new users)

1. Create the environment from the provided specification:
   ```bash
   conda env create -f environment.yml
   ```
2. Activate it:
   ```bash
   conda activate optilb
   ```
3. Confirm the editable installation succeeded:
   ```bash
   python -c "import optilb; print(optilb.__version__)"
   ```
4. (Optional) enable the NOMAD optimiser:
   ```bash
   pip install '.[nomad]'
   ```
5. (Optional) update the environment after pulling new changes:
   ```bash
   conda env update -f environment.yml --prune
   ```

*What you get*: Python 3.11 (from `environment.yml`), NumPy, SciPy, matplotlib,
`pytest`, linting/type-checking tooling, and an editable install of `optilb`
with examples ready to run.

---

## 4. Option B - Native Python virtual environment

Use this path if you prefer to manage dependencies with the standard library or
`pipx`.

1. Ensure `pip` is current:
   ```bash
   python -m pip install --upgrade pip
   ```
2. Create and activate a virtual environment (example uses `.venv`):
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   # .venv\Scripts\activate.bat     # Windows (cmd)
   # .venv\Scripts\Activate.ps1     # Windows (PowerShell)
   ```
3. Install runtime dependencies and the package in editable mode with helpful extras:
   ```bash
   pip install '.[examples,dev]'
   ```
4. (Optional) enable the NOMAD optimiser:
   ```bash
   pip install '.[nomad]'
   ```
5. Verify the import:
   ```bash
   python -c "import optilb; print(optilb.__version__)"
   ```

Deactivate the virtual environment with `deactivate` when you are done.

---

## 5. Sanity checks

Run a quick test suite to confirm everything is wired correctly:

```bash
pytest -q
```

You can also execute an example script and inspect generated outputs:

```bash
python examples/parallel_speedup.py --help
```

For a simple smoke test of the optimisation facade:

```bash
python -c "from optilb import OptimizationProblem, DesignSpace, get_objective; \
space = DesignSpace(lower=[-5, -5], upper=[5, 5]); \
obj = get_objective('quadratic'); \
print(OptimizationProblem(obj, space, [0, 0], optimizer='bfgs').run().best_f)"
```

---

## 6. Upgrading or reinstalling

- **Conda environment**: `conda env update -f environment.yml --prune`
- **Virtual environment**: Re-run `pip install '.[examples,dev]'` (and
  optionally `'.[nomad]'`) after pulling changes.

If you encounter conflicts, consider wiping the environment and recreating it.

---

## 7. Troubleshooting tips

- Ensure the active Python interpreter matches the environment you intend to
  use (`which python` or `where python`).
- If `pip` cannot find wheels for `PyNomadBBO`, either install a supported
  Python version (>= 3.10) or skip the `nomad` extra; the MADS optimiser will
  raise `MissingDependencyError` if invoked without the dependency.
- If matplotlib fails to import due to backend issues, set the Agg backend
  before running scripts:
  ```bash
  export MPLBACKEND=Agg
  ```
- Windows users seeing "Scripts is not on PATH" warnings likely skipped the
  activation step. Launch a new shell, run the activation command appropriate
  for your shell, and retry.

With either workflow configured, you can modify the source under `src/` and
rerun examples or tests without reinstalling.
