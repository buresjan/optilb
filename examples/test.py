import numpy as np
import matplotlib.pyplot as plt
from optilb import DesignSpace
from optilb.optimizers import BFGSOptimizer, NelderMeadOptimizer, MADSOptimizer

# Define a 2D rugged test function (continuous and piecewise-constant versions)
def rugged_objective(x: np.ndarray, continuous: bool = True, scale: float = 0.15) -> float:
    """Rugged multimodal function with smooth or quantized output.
    - continuous=True: returns a smooth value (mimics interpolated boundary).
    - continuous=False: returns a piecewise-constant value with cell size `scale` (mimics bounce-back)."""
    # Base continuous function: mix of sin waves and a parabola (multimodal, smooth)
    x1, x2 = x  # two parameters
    smooth_val = np.sin(3*x1) * np.sin(3*x2) + 0.1 * (x1**2 + x2**2)
    if continuous:
        return smooth_val
    else:
        # Quantize x1, x2 to nearest grid center of size `scale`
        x1q = (np.floor(x1/scale) + 0.5) * scale
        x2q = (np.floor(x2/scale) + 0.5) * scale
        return np.sin(3*x1q) * np.sin(3*x2q) + 0.1 * (x1q**2 + x2q**2)
    
# Create a grid
x = np.linspace(-2, 2, 200)
y = np.linspace(-2, 2, 200)
X, Y = np.meshgrid(x, y)

# Evaluate both versions
Z_continuous = np.vectorize(lambda a, b: rugged_objective((a, b), continuous=True))(X, Y)
Z_constant   = np.vectorize(lambda a, b: rugged_objective((a, b), continuous=False))(X, Y)

# Plot side-by-side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Continuous
im1 = axes[0].imshow(Z_continuous, extent=[x.min(), x.max(), y.min(), y.max()],
                     origin='lower', cmap='viridis')
axes[0].set_title("Continuous version")
fig.colorbar(im1, ax=axes[0])

# Piecewise-constant
im2 = axes[1].imshow(Z_constant, extent=[x.min(), x.max(), y.min(), y.max()],
                     origin='lower', cmap='viridis')
axes[1].set_title("Piecewise-constant version")
fig.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.show()

# Set up a 2D design space for the parameters (e.g., both in [-1.5, 1.5])
space = DesignSpace(lower=[-1.5, -1.5], upper=[1.5, 1.5])

# Initial guess for optimization
x0 = np.array([0.001, -0.001])

# Create optimizer instances (using 4 workers for demonstration of parallel capability)
opt_bfgs = BFGSOptimizer(n_workers=4)           # gradient-based (will use finite diffs)
opt_nm   = NelderMeadOptimizer(n_workers=4)     # Nelder-Mead simplex
try:
    opt_mads = MADSOptimizer(n_workers=4)       # MADS (needs PyNomadBBO installed)
except ImportError:
    opt_mads = None

# Run BFGS on the continuous (smooth) objective
res_bfgs_cont = opt_bfgs.optimize(lambda x: rugged_objective(x, continuous=True),
                                  x0, space, max_iter=100, tol=1e-6, parallel=True)
print(f"BFGS on continuous: f_min = {res_bfgs_cont.best_f:.4f} at x = {res_bfgs_cont.best_x},  evals = {res_bfgs_cont.nfev}")

# Run BFGS on the piecewise-constant objective (expected to struggle)
res_bfgs_disc = opt_bfgs.optimize(lambda x: rugged_objective(x, continuous=False, scale=0.15),
                                  x0, space, max_iter=100, tol=1e-6, parallel=True)
print(f"BFGS on piecewise-constant: f_min = {res_bfgs_disc.best_f:.4f} at x = {res_bfgs_disc.best_x},  evals = {res_bfgs_disc.nfev}")


# Run MADS on the piecewise-constant objective (if available)
if opt_mads:
    res_mads = opt_mads.optimize(lambda x: rugged_objective(x, continuous=False, scale=0.15),
                                 x0, space, max_iter=200, tol=1e-6, parallel=True)
    print(f"MADS on piecewise-const: f_min = {res_mads.best_f:.4f} at x = {res_mads.best_x},  evals = {res_mads.nfev}")
else:
    print("MADS optimizer not available (PyNomadBBO not installed).")


# Run Nelder–Mead on the piecewise-constant objective
res_nm = opt_nm.optimize(lambda x: rugged_objective(x, continuous=False, scale=0.1),
                         x0, space, max_iter=200, tol=1e-6, parallel=True)
print(f"Nelder-Mead on piecewise-const: f_min = {res_nm.best_f:.4f} at x = {res_nm.best_x},  evals = {res_nm.nfev}")
