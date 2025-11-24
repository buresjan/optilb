# Changelog

## Unreleased
- Hardened MADS normalization: preserves constraint metadata, validates bounds early, records history in original space, and clears scaled history between runs.
- Expanded documentation for core data structures, optimiser options, the
  optimisation façade, and scheduled runs.
- Nelder–Mead now accepts a predefined simplex plus objective values
  (`initial_simplex`/`initial_simplex_values`), optionally normalised to the
  unit cube, skipping redundant initial evaluations.
