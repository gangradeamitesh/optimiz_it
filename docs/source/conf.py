# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../Optimizer/optimiz_it'))

project = 'optimiz'
copyright = '2024, Amitesh Gangrade'
author = 'Amitesh Gangrade'
release = '0.2.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
       'sphinx.ext.autodoc',  # Automatically document from docstrings
       'sphinx.ext.viewcode',  # Add links to highlighted source code
       'sphinx.ext.napoleon',  # Support for Google style docstrings
]
templates_path = ['_templates']
exclude_patterns = []

language = 'Python'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
