from __future__ import annotations
from datetime import datetime

# -- Project info -------------------------------------------------------------
project = "Aarambam"
author = "Dhayaa Anbajagane"
copyright = f"{datetime.now():%Y}, {author}"
html_theme = "furo"

# -- Extensions ---------------------------------------------------------------
extensions = [
    "myst_parser",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "numpydoc",
    "autoapi.extension",      # parse source without importing
]

# Accept both Markdown and reST sources
source_suffix = [".md", ".rst"]

# -- MyST options -------------------------------------------------------------
myst_enable_extensions = ["colon_fence", "deflist", "tasklist", "linkify"]

# -- Intersphinx --------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
}

# -- AutoAPI (relative to THIS conf.py) --------------------------------------
# conf.py is at docs/source/, package is at ../../Aarambam
autoapi_type = "python"
autoapi_dirs = ["../../Aarambam"]
autoapi_root = "api"                      # generated under /api/
autoapi_add_toctree_entry = True          # adds "API Reference" to TOC
autoapi_keep_files = True                 # keep generated rst for debugging
autoapi_options = [
    "members",
    "undoc-members",
    "show-inheritance",
    "show-module-summary",
    "imported-members",
]
autoapi_ignore = ["*/tests/*", "*/test_*", "*_version.py"]
autoapi_python_class_content = "both"

# -- Numpydoc -----------------------------------------------------------------
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

# -- Build hygiene ------------------------------------------------------------
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Important: do NOT import your package here.
