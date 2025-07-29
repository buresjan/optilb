# Documentation Overview

The documentation is written in both reStructuredText (.rst) and Markdown (.md) formats.  Keep both versions in sync when new features are added.

* `sampling.rst` and `sampling.md` cover the Latin-Hypercube sampler.
* `api/` contains API reference files like `sampling.rst` and `sampling.md`.

The Sphinx build currently uses the `.rst` files, but Markdown copies are provided for quick browsing on GitHub.

To generate the HTML documentation locally::

    sphinx-build -b html docs docs/_build/html
