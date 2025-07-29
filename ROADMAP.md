# ROADMAP.md
---

## 0. Vision Recap
**`optilb`** aims to be a **low‑dimension shape‑optimisation toolkit** that:

| Pillar | Key Points |
|--------|------------|
| **Sampling** | Latin‑Hypercube (space‑filling, reproducible). |
| **Local Search** | Mesh Adaptive Direct Search (PyNOMAD), L‑BFGS‑B (SciPy), custom parallel Nelder‑Mead. |
| **CFD Coupling** | Plug‑n‑play: analytic toy functions → LBM stub → GPU‑LBM executable. |
| **Surrogates** | Optional Gaussian‑Process with EI acquisition. |
| **Uncertainty** | Monte‑Carlo / Sobol sampling of boundary‑condition distributions; Reliability‑Based Robust Optimisation (RBRO). |
| **Unified API** | One façade (`OptimizationProblem`) controlling the pipeline. |

---

## 1. Milestone Breakdown

| # | Title | Status | Target Date |
|---|-------|--------|-------------|
| 1 | Repo scaffold & tooling | ☐ |
| 2 | Core dataclasses | ☐ |
| 3 | LHS sampler | ☐ |
| 4 | Analytic objectives | ☐ |
| 5 | LBM stub objective | ☐ |
| 6 | External CFD wrapper | ☐ |
| 7 | Base optimiser interface | ☐ |
| 8 | SciPy BFGS wrapper | ☐ |
| 9 | PyNOMAD MADS wrapper | ☐ |
| 10 | Parallel Nelder‑Mead | ☐ |
| 11 | Early‑stopping utility | ☐ |
| 12 | UQ module | ☐ |
| 13 | GP surrogate | ☐ |
| 14 | Robust optimiser | ☐ |
| 15 | High‑level façade | ☐ |
| 16 | Coverage & CI hardening | ☐ |
| 17 | Example 1 notebook | ☐ |
| 18 | Example 2 notebook | ☐ |
| 19 | Example 3 notebook | ☐ |
| 20 | Documentation site | ☐ |

> **Legend:** ☐ not started · ◑ in progress · ✓ done

---

## 2. Detailed Task Descriptions
Full specifications live in `TASKS.md` (generated from the conversation).  
Agents **must read the relevant task before coding**.  
Each task block contains:

- **Objective** – single‑line goal.  
- **Context** – why we need it.  
- **Requirements** – acceptance criteria.  
- **Inputs / Outputs** – affected files & APIs.  
- **Notes** – style or performance hints.

---

## 3. Stretch Goals (post‑v1)
This is not important now.
1. **Bayesian Optimisation front‑end** (BoTorch).  
2. **Multi‑fidelity framework** combining stub & high‑fidelity CFD.  
3. **Dashboard** using Dash/Plotly to monitor optimisation live.  
4. **Automatic differentiation support** (JAX) for gradient‑based modes.  
5. **Distributed batch evaluation** on Kubernetes with Ray Serve.

---

## 4. Technical Debt & Risks
Not necessary now.
| Area | Risk | Mitigation |
|------|------|------------|
| PyNOMAD wheels | Upstream may lag for new Python versions. | Vendor minimal C bindings if project stalls. |
| GPU‑LBM binary | Hardware‑specific & proprietary. | Keep stub + external CLI interface to swap binaries. |
| GP scaling | O(n³) memory in >1 k samples. | Limit dims to <10; add sparse‑GP in Stretch Goals. |

---

## 5. Contribution Workflow
1. **Find the task closest to what you are doing** (see table above).  
2. Create branch `codex‑XX‑<short‑slug>`.  
3. Follow coding, testing and doc rules in `AGENTS.md`.  
4. Open PR; ensure CI passes; request review.  
5. After merge, **update the status box in this file** (replace ☐ → ✓).

---

## 6. Release Plan
| Version | Contents | ETA |
|---------|----------|-----|
| **v0.1.0** | Tasks 1‑10 (basic optimisation loop + toy objectives) | 10 Aug 2025 |
| **v0.2.0** | Tasks 11‑15 (UQ, surrogate, façade) | 20 Aug 2025 |
| **v0.3.0** | Tasks 16‑19 (examples, CI hardening) | 31 Aug 2025 |
| **v1.0.0** | Task 20 (docs site) + any bug fixes | 07 Sep 2025 |

Semantic‑versioning: MAJOR API break, MINOR feature, PATCH bug‑fix.

---

## 7. Useful Links
- Reference thesis (PDF) → `/docs/refs/bures‑thesis.pdf`
- Inspiration paper `optim.pdf` → `/docs/refs/optim.pdf`
- PyNOMAD GitHub → <https://github.com/bbopt/PyNomad>
- SciPy minimise docs → <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>

---

*Stay aligned with this roadmap—update it whenever scope moves.*
