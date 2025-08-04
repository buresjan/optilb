# Documentation Overview

The documentation is written in both reStructuredText (.rst) and Markdown (.md) formats.  Keep both versions in sync when new features are added.

* `sampling.rst`/`sampling.md` cover the Latin-Hypercube sampler.
* `objectives.rst`/`objectives.md` document analytic objective functions.
* `optimizers.rst`/`optimizers.md` describe built-in local optimizers.
* `api/` contains API reference files mirroring these modules.

The Sphinx build currently uses the `.rst` files, but Markdown copies are provided for quick browsing on GitHub.

To generate the HTML documentation locally::

    sphinx-build -b html docs docs/_build/html
