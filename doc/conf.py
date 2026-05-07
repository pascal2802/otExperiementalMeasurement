# -*- coding: utf-8 -*-
#
# otGpxTrack documentation build configuration file.
#
import os
import sys
import subprocess  # noqa: F401 (peut rester même si non utilisé désormais)

# Ajoute le projet au PYTHONPATH pour autodoc
sys.path.insert(0, os.path.abspath("../"))

# -- Extensions Sphinx ---------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "numpydoc",
    "sphinx_gallery.gen_gallery",
    # Remplacement d'imgmath par mathjax (pas de dépendance LaTeX pour HTML)
    "sphinx.ext.mathjax",
]

sphinx_gallery_conf = {
    'examples_dirs': ['examples'],
    'gallery_dirs': ['auto_examples'],
    'show_signature': False,
}

autodoc_default_options = {
    "members": None,
    "inherited-members": None,
    "exclude-members": "thisown",
}

# --- Configuration MathJax (HTML) --------------------------------------------
# Par défaut, Sphinx charge MathJax v3 depuis un CDN.
# Si vos runners n'ont pas accès à Internet, déposez une copie locale et
# définissez par exemple :
# mathjax_path = "_static/tex-mml-chtml.js"
# mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

mathjax3_config = {
    "tex": {
        "tags": "ams",
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "packages": {"[+]": ["base", "ams", "noerrors", "noundefined"]},
        # Décommentez si besoin de macros :
        # "macros": { "RR": "\\mathbb{R}" }
    }
}

# -- Intersphinx ---------------------------------------------------------------
intersphinx_mapping = {
    "python": ("http://openturns.github.io/openturns/1.26", "openturns-objects.inv")
}

# -- Numpydoc / Autosummary ----------------------------------------------------
autosummary_generate = True
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False

# -- Templates / Sources -------------------------------------------------------
templates_path = ["_templates"]
source_suffix = [".rst"]
master_doc = "index"

# -- Infos projet --------------------------------------------------------------
project = u"otGpxTrack"
copyright = u"2005-2024 Airbus-EDF-IMACS-ONERA-Phimeca"
version = "0.1"
release = "0.1"

# -- Divers --------------------------------------------------------------------
exclude_patterns = []
add_module_names = False
pygments_style = "friendly"
todo_include_todos = True

# -- HTML ----------------------------------------------------------------------
html_theme = "openturns"
html_sourcelink_suffix = ""
html_theme_path = ["themes"]
html_sidebars = {
    "**": ["globaltoc.html", "relations.html", "sourcelink.html", "searchbox.html"]
}
html_static_path = ["_static"]
html_last_updated_fmt = "%b %d, %Y"
html_show_sourcelink = True

# -- LaTeX (pour build PDF uniquement) ----------------------------------------
latex_preamble = r"""
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{expdlist}
\usepackage{math_notations}
\usepackage{stackrel}
\let\latexdescription=\description
\def\description{\latexdescription{}{} \breaklabel}
\DeclareMathOperator*{\argmin}{Argmin}
"""

latex_elements = {
    "papersize": "a4paper",
    "pointsize": "10pt",
    "preamble": latex_preamble,
}

latex_documents = [
    (
        "index",
        "otGpxTrack.tex",
        u"otGpxTrack Documentation",
        u"Airbus-EDF-IMACS-Phimeca-ONERA",
        "manual",
    ),
]

# -- Man page ------------------------------------------------------------------
man_pages = [
    (
        "index",
        "otGpxTrack",
        u"otGpxTrack Documentation",
        [u"Airbus-EDF-IMACS-Phimeca-ONERA"],
        1,
    )
]

# -- Texinfo -------------------------------------------------------------------
texinfo_documents = [
    (
        "index",
        "otGpxTrack",
        u"otGpxTrack Documentation",
        u"Airbus-EDF-IMACS-Phimeca-ONERA",
        "otGpxTrack",
        "One line description of project.",
        "Miscellaneous",
    ),
]

