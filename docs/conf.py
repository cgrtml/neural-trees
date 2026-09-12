"""Sphinx configuration for the neural-trees documentation."""

import os
import sys
from datetime import date

sys.path.insert(0, os.path.abspath(".."))

import neural_trees  # noqa: E402

project = "neural-trees"
author = "Cagri Temel"
copyright = f"{date.today().year}, {author}"
version = neural_trees.__version__
release = version

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    "myst_parser",
]

autosummary_generate = True
autodoc_default_options = {"members": True, "inherited-members": False}
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
    "torch": ("https://docs.pytorch.org/docs/stable", None),
}

sphinx_gallery_conf = {
    "examples_dirs": "../examples",
    "gallery_dirs": "auto_examples",
    "filename_pattern": r"\.py",
    "ignore_pattern": r"__init__\.py",
    "remove_config_comments": True,
    "download_all_examples": False,
    "plot_gallery": "True",
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
source_suffix = {".rst": "restructuredtext", ".md": "markdown"}

html_theme = "pydata_sphinx_theme"
html_title = f"neural-trees {version}"
html_static_path = ["_static"]
html_theme_options = {
    "github_url": "https://github.com/cgrtml/neural-trees",
    "icon_links": [
        {
            "name": "PyPI",
            "url": "https://pypi.org/project/neural-trees/",
            "icon": "fa-brands fa-python",
        },
    ],
    "show_prev_next": False,
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
}
html_context = {"default_mode": "auto"}
