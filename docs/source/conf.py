# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------

project = 'oc_py'
copyright = '2026, Barış GÜLER, Özgür BAŞTÜRK, Mohammad NIAEI'
author = 'Barış GÜLER, Özgür BAŞTÜRK, Mohammad NIAEI'
release = '0.0.1b1'

# Make sure the package source is importable by autodoc
sys.path.insert(0, os.path.abspath('../../src'))

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------

extensions = [
    'myst_nb',                 # Jupyter notebook + Markdown support (includes myst_parser)
    'sphinx.ext.autodoc',      # Generate docs from docstrings
    'sphinx.ext.napoleon',     # Parse NumPy / Google style docstrings
    'sphinx.ext.viewcode',     # Add [source] links to API pages
    'sphinx.ext.autosummary',  # Generate summary tables
]

# ---------------------------------------------------------------------------
# Source file handling
# ---------------------------------------------------------------------------

source_suffix = {
    '.rst': 'restructuredtext',
    # .md and .ipynb are registered automatically by myst_nb
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# ---------------------------------------------------------------------------
# MyST (Markdown / notebook) settings
# ---------------------------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",   # ::: fences as directives
    "deflist",       # definition lists
    "dollarmath",    # $...$ and $$...$$ math
]

nb_execution_mode = "off"   # never execute notebooks during build

# ---------------------------------------------------------------------------
# autodoc settings
# ---------------------------------------------------------------------------

autodoc_typehints = "description"            # put type hints in param descriptions
autodoc_typehints_description_target = "documented"
autodoc_member_order = "bysource"            # preserve source order
autodoc_default_options = {
    "members":          True,
    "undoc-members":    True,
    "show-inheritance": True,
}

# ---------------------------------------------------------------------------
# Napoleon (NumPy / Google docstring) settings
# ---------------------------------------------------------------------------

napoleon_google_docstring  = True
napoleon_numpy_docstring   = True
napoleon_use_param         = True   # emit :param: roles (required for autodoc_typehints)
napoleon_use_rtype         = True   # emit :rtype: role
napoleon_preprocess_types  = True
napoleon_attr_annotations  = True

# ---------------------------------------------------------------------------
# Suppress known harmless warnings
# ---------------------------------------------------------------------------

suppress_warnings = [
    "ref.duplicate",   # duplicate object descriptions across rst files
]

# ---------------------------------------------------------------------------
# HTML output
# ---------------------------------------------------------------------------

html_theme = 'furo'

html_theme_options = {
    "navigation_with_keys": True,
    "sidebar_hide_name":    False,
}

html_static_path = ['_static']
