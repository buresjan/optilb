# optilb

**optilb** is an upcoming optimisation toolbox for low-dimensional shape tuning.
The roadmap outlines a workflow that combines Latin-Hypercube sampling with
pluggable optimisers (Mesh Adaptive Direct Search via PyNOMAD, SciPy's
L-BFGS-B and a custom parallel Nelder–Mead).  CFD objectives—from analytic
toy cases to external executables—will be wrapped through a unified
`OptimizationProblem` interface.  Optional Gaussian‑Process surrogates and
robust optimisation utilities are scheduled for later milestones.

This repository currently contains only the scaffolding, but it will grow as the
roadmap tasks are tackled.  See `ROADMAP.md` for planned features and progress.
