from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "optilb"
author = "Jan Bures"
release = "0.0.0"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

exclude_patterns: list[str] = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
