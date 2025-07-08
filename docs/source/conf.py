# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
import os.path

basedir = os.path.abspath(os.path.join(pathlib.Path(__file__).parents[2], "src"))
sys.path.insert(0, basedir)

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Librosax"
copyright = "2025, David Braun"
author = "David Braun"
first_line = open(
    os.path.join(pathlib.Path(__file__).parents[2], "src/librosax/version.py"), "r"
).readline()
# first_line is 'version = "1.2.3"'
assert first_line.startswith("version = ")
release = first_line.split("=")[1].strip()[1:-1]

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
]

# Autosummary settings
autosummary_generate = True
autodoc_member_order = 'bysource'

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = False

# Intersphinx mappings
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'jax': ('https://jax.readthedocs.io/en/latest/', None),
}

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
html_theme = "furo"
html_static_path = ["_static"]
html_title = f"Librosax Documentation, v{release}"

add_module_names = False
autoclass_signature = "separated"
todo_include_todos = True
napoleon_use_ivar = True

html_theme_options = {
    "top_of_page_buttons": [],
}
