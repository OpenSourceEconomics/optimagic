#!/usr/bin/env python3
#
# optimagic documentation build configuration file, created by
# sphinx-quickstart on Fri Jan 18 10:59:27 2019.
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime as dt
import os
from importlib.metadata import version

year = dt.datetime.now().year

author = "Janos Gabler"

# Set variable so that todos are shown in local build
on_rtd = os.environ.get("READTHEDOCS") == "True"


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx_copybutton",
    "myst_nb",
    "sphinxcontrib.bibtex",
    "sphinx_panels",
    "sphinx_design",
]

myst_enable_extensions = [
    "colon_fence",
    "dollarmath",
    "html_image",
]
copybutton_prompt_text = ">>> "
copybutton_only_copy_prompt_lines = False

bibtex_bibfiles = ["refs.bib"]

autodoc_member_order = "bysource"

autodoc_mock_imports = [
    "bokeh",
    "cloudpickle",
    "cyipopt",
    "fides",
    "joblib",
    "nlopt",
    "pytest",
    "pygmo",
    "scipy",
    "sqlalchemy",
    "tornado",
    "petsc4py",
    "statsmodels",
    "numba",
]

extlinks = {
    "ghuser": ("https://github.com/%s", "@"),
    "gh": ("https://github.com/optimagic-dev/optimagic/pulls/%s", "#"),
}

intersphinx_mapping = {
    "numpy": ("https://docs.scipy.org/doc/numpy", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3.12", None),
}

linkcheck_ignore = [
    r"https://tinyurl\.com/*.",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
source_suffix = [".rst", ".ipynb", ".md"]

# The master toctree document.
master_doc = "index"

# General information about the project.
project = "optimagic"
copyright = f"2019 - {year}, {author}"  # noqa: A001

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
release = version("optimagic").split("+")[0]
version = ".".join(release.split(".")[:2])

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ["_build", "**.ipynb_checkpoints"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"
pygments_dark_style = "monokai"

# If true, `todo` and `todoList` produce output, else they produce nothing.
if on_rtd:
    pass
else:
    todo_include_todos = True
    todo_emit_warnings = True

# -- Options for myst-nb  ----------------------------------------
nb_execution_mode = "force"
nb_execution_allow_errors = False
nb_merge_streams = True

# Notebook cell execution timeout; defaults to 30.
nb_execution_timeout = 1000

# List of notebooks that will not be executed.
nb_execution_excludepatterns = [
    # Problem with latex rendering
    "estimation_tables_overview.ipynb",
    # too long runtime
    "bootstrap_montecarlo_comparison.ipynb",
]

# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "furo"

# Add any paths that contain custom static files (such as style sheets) here, relative
# to this directory. They are copied after the built-in static files, so a file named
# "default.css" will overwrite the built-in "default.css".
html_css_files = ["css/termynal.css", "css/termynal_custom.css", "css/custom.css"]

html_js_files = ["js/termynal.js", "js/custom.js"]


# Add any paths that contain custom static files (such as style sheets) here, relative
# to this directory. They are copied after the builtin static files, so a file named
# "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If false, no module index is generated.
html_domain_indices = True

# If false, no index is generated.
html_use_index = True

# If true, the index is split into individual pages for each letter.
html_split_index = False

# If true, links to the reST sources are added to the pages.
html_show_sourcelink = False

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = True

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

html_title = "optimagic"

html_theme_options = {
    "sidebar_hide_name": True,
    "navigation_with_keys": True,
    "light_logo": "images/optimagic_logo.svg",
    "dark_logo": "images/optimagic_logo_dark_mode.svg",
    "light_css_variables": {
        "color-brand-primary": "#f04f43",
        "color-brand-content": "#f04f43",
    },
    "dark_css_variables": {
        "color-brand-primary": "#f04f43",
        "color-brand-content": "#f04f43",
    },
}
