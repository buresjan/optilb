# AGENTS.md

## ‟Why this file exists”
This document tells **all autonomous or semi‑autonomous developer agents** (e.g. OpenAI Codex‑developer, CI bots, doc builders) **how to behave inside the *optilb* repository**.  
It is the single source of truth for day‑to‑day conventions.  
Every agent **must also read `ROADMAP.md`** to understand the big‑picture vision and upcoming milestones.

---

## 1 — Mission Charter
1. **Deliver a fully Pythonic optimisation framework (`optilb`)** that can:
   - generate Latin‑Hypercube samples,
   - run pluggable local optimisers (MADS via PyNOMAD, L‑BFGS‑B via SciPy, custom parallel Nelder‑Mead),
   - wrap CFD/LBM objectives,
   - (optionally) build Gaussian‑Process surrogates,
   - quantify boundary‑condition uncertainty,
   - expose everything through a unified `OptimizationProblem` API.
2. Keep the repository **easy to install, easy to read, and easy to extend**.

---

## 2 — How to pick up work
1. **Open `ROADMAP.md`**  
   – The roadmap lists numbered tasks in the order they should be tackled.  
2. Search the repo for an *open* or *partially complete* task directory / test.  
3. Create or update code **only inside the scope of the current task**. If you discover a prerequisite that is missing, add a *sub‑task* comment to `ROADMAP.md` instead of silently expanding the task.

---

## 3 — Repository Conventions
| Area | Rule |
|------|------|
| **Layout** | Source lives in `src/optilb/`; tests in `tests/`; notebooks in `examples/`; docs in `docs/`. |
| **Packaging** | PEP 517 via `pyproject.toml`; no `setup.py`. |
| **Python ver.** | 3.10 ≤ version ≤ 3.12. |
| **Style** | `black`, `isort`, `flake8`, `mypy`; run `pre‑commit run --all-files` before committing. |
| **Typing** | Use *PEP 563 postponed annotations* and `from __future__ import annotations`. |
| **Imports** | Absolute inside `optilb.*`; relative (`.`) only inside sub‑packages. |
| **Logging** | Use the project‑wide logger obtained via `logging.getLogger("optilb")`. |
| **Errors** | Prefer custom exceptions under `optilb.exceptions` to `RuntimeError`. |
| **Dependencies** | Keep `requirements.txt` updated whenever a new dependency is added. |

---

## 4 — Commit & PR etiquette
1. **Branch name:** `task‑XX‑<slug>`, where `XX` refers to the ROADMAP task number.  
2. **Commit message:**  

3. **Pull requests must:**
- link the task in ROADMAP (`task‑XX`),
- pass CI (lint + tests + coverage ≥ 80 %),
- update docs if public API changed.

---

## 5 — Testing & CI
- Use `pytest -q` locally; short tests *must* run in < 30 s.
- Long‑running examples are kept in `examples/` and excluded from default test run via `pytest --ignore examples`.
- **Coverage goal:** ≥ 80 % lines; enforced by the GitHub Actions workflow.

---

## 6 — Documentation workflow
1. Docstrings use **Google style**.  
2. Every public symbol appears in `docs/api/`.  
3. After merging to `main`, CI auto‑builds HTML with `sphinx -W` (warnings as errors) and deploys to GitHub Pages.
4. Keep `README.md` updated whenever features are added or changed.
5. For each new functionality, add a dedicated documentation file under `docs/` describing it in detail.

---

## 7 — Security & resource limits
- No network access at runtime except when explicitly whitelisted in task spec (e.g. pulling remote CFD results).  
- GPU usage must be **optional** and feature‑gated.  
- Large binary artefacts (> 20 MB) go to git‑LFS or an external release, *not* the main repo.

---

## 8 — When in doubt
1. **Re‑read the corresponding section in `ROADMAP.md`.**  
2. If still unclear, open an issue tagged `clarification`.  
3. Do not implement speculative features—stick to the roadmap.

*Happy hacking, agents!*
