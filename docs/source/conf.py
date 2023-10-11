# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import sys, os
sys.path.insert(0, os.path.abspath('../../mika/'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MIKA'
copyright = '2023, Hannah Walsh and Sequoia Andrade'
author = 'Hannah Walsh and Sequoia Andrade'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc', 'sphinx.ext.napoleon', 'nbsphinx', 'myst_parser']

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
