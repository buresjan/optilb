# AGENTS.md

## Why this file exists
This document tells all autonomous or semi-autonomous developer agents (for example OpenAI Codex, CI bots, doc builders) how to behave inside the *optilb* repository.
It is the single source of truth for day-to-day conventions.
For high-level orientation, also read `README.md`, the docs entry point in `docs/index.rst`, and any issue or pull request that defines your task.

---

## 1 - Mission charter
1. Deliver a fully Pythonic optimisation framework (`optilb`) that can:
   - generate Latin-Hypercube samples,
   - run pluggable local optimisers (MADS via PyNOMAD, L-BFGS-B via SciPy, custom parallel Nelder-Mead),
   - wrap CFD/LBM objectives,
   - optionally build Gaussian-Process surrogates,
   - quantify boundary-condition uncertainty,
   - expose everything through a unified `OptimizationProblem` API.
2. Keep the repository easy to install, easy to read, and easy to extend.

---

## 2 - Work intake
1. Start with the GitHub issue tracker or project board. Items labelled `task`, `bug`, or `enhancement` describe the current backlog and must be linked in your branch or pull request.
2. Open the relevant documentation under `docs/` (for example `docs/core.md`, `docs/optimizers.md`, `docs/objectives.md`) and the existing tests in `tests/` to confirm expected behaviour before making changes.
3. Implement code only within the agreed scope. If you discover a missing prerequisite, add an issue tagged `clarification` or comment on the existing ticket instead of silently expanding the task.
4. Prefer incremental work: land the smallest meaningful slice that moves the linked issue forward, and keep future ideas in follow-up tickets.

---

## 3 - Agentic operations tips
- Begin by restating the problem, identifying impacted modules, and collecting context with `rg`, `pytest -k`, and `python -m pip show` when needed.
- Produce a brief plan before editing; update it as steps complete so collaborators can follow your reasoning.
- Keep diffs tight: prefer surgical edits, add comments only when they unlock understanding, and avoid rewrites unless the issue demands them.
- Run the most targeted checks possible (`pytest -k`, `mypy`, formatters) before proposing a change; explain any tests you could not execute.
- Record assumptions in commit messages and PR descriptions so humans can audit automated work.
- Leave the repository ready for the next agent: ensure docs, type hints, and requirements stay in sync with the code you touch.

---

## 4 - Repository conventions
| Area | Rule |
|------|------|
| Layout | Source lives in `src/optilb/`; tests in `tests/`; notebooks in `examples/`; docs in `docs/`. |
| Packaging | PEP 517 via `pyproject.toml`; no `setup.py`. |
| Python ver. | 3.10 ≤ version ≤ 3.12. |
| Style | `black`, `isort`, `flake8`, `mypy`. |
| Typing | Use postponed annotations via `from __future__ import annotations`. |
| Imports | Absolute inside `optilb.*`; relative (`.`) only inside sub-packages. |
| Logging | Use the project-wide logger obtained via `logging.getLogger("optilb")`. |
| Errors | Prefer custom exceptions under `optilb.exceptions` to `RuntimeError`. |
| Dependencies | Keep `requirements.txt` updated whenever a new dependency is added. |

---

## 5 - Commit and PR etiquette
1. Branch name: `issue-<number>-<slug>` (for example `issue-102-fix-lhs`) or `topic/<slug>` when no issue exists yet.
2. Commit message: short imperative summary (`Add fast Nelder-Mead test`), with a body that lists motivation, key changes, and any follow-up TODOs.
3. Pull requests must:
   - reference the driving issue or ticket,
   - pass CI (lint + tests + coverage ≥ 80%),
   - update docs whenever the public API or behaviour changes.

---

## 6 - Testing and CI
- Use `pytest -q` locally; short tests must run in under 30 seconds.
- Long-running examples live in `examples/` and are excluded from the default run via `pytest --ignore examples`.
- Coverage goal: ≥ 80% lines; enforced by the GitHub Actions workflow.

---

## 7 - Documentation workflow
1. Docstrings use Google style.
2. Every public symbol appears in `docs/api/`.
3. After merging to `main`, CI auto-builds HTML with `sphinx -W` (warnings as errors) and deploys to GitHub Pages.
4. Keep `README.md` updated whenever features are added or changed.
5. For each new functionality, add a dedicated documentation file under `docs/` describing it in detail.
6. Provide both `.rst` and `.md` versions of each documentation page so that Sphinx and GitHub viewers stay in sync.
7. Documentation must include a short usage snippet so new tools and agents can replicate the feature easily. Pull requests lacking accompanying docs are rejected.

---

## 8 - Security and resource limits
- No network access at runtime except when explicitly whitelisted (for example pulling remote CFD results).
- GPU usage must be optional and feature-gated.
- Large binary artefacts (> 20 MB) go to git-LFS or an external release, not the main repo.

---

## 9 - When in doubt
1. Re-read the relevant issue, this guide, and any module-level docs under `docs/`.
2. If still unclear, open an issue tagged `clarification`.
3. Do not implement speculative features—stick to the agreed backlog.

Happy hacking, agents!
