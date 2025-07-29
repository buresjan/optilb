"""Basic sanity checks for the optilb package.

Run this module from the repository root with:

    PYTHONPATH=./src python examples/basic_usage.py

It should execute without printing anything if all assertions pass.
"""

from optilb.core import DesignPoint, DesignSpace
from optilb.sampling import lhs
from optilb.objectives import get_objective

# --- core dataclasses -------------------------------------------------------

ds = DesignSpace(lower=[-1, -1], upper=[1, 1])
pt = DesignPoint(x=[0.3, 0.7])
assert pt.x[1] == 0.7 and ds.dimension == 2

# --- sampling ---------------------------------------------------------------

pts = lhs(5, DesignSpace(lower=[0, 0, 0], upper=[1, 1, 1]), seed=42)
assert all(0.0 <= x <= 1.0 for p in pts for x in p.x)

ds_mixed = DesignSpace(lower=[0, 0], upper=[10, 1])
pts = lhs(20, ds_mixed)
assert all(p.x[0] == int(p.x[0]) for p in pts)

# --- objectives -------------------------------------------------------------

f = get_objective("rastrigin")
assert round(f([0, 0]), 5) == 0

g = get_objective("noisy_discontinuous", sigma=0.0)
assert g([1, 1]) == g([1, 1])

print("All example assertions passed.")
