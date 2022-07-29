# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys

import warnings
from datetime import date

import sorts

import os
import sys
import importlib

sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------

project = 'SORTS'
version ='.'.join(sorts.__version__.split('.')[:2])
release = sorts.__version__

copyright = f'[2020-{date.today().year}] Daniel Kastinen, Juha Vierinen, Thomas Maynadie'
author = 'Daniel Kastinen, Juha Vierinen, Thomas Maynadie'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "numpydoc",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    'sphinx.ext.mathjax',
    'sphinx_panels',
    ]

skippable_extensions = [
    ('breathe', 'skip generating C/C++ API from comment blocks.'),
]
for ext, warn in skippable_extensions:
    ext_exist = importlib.util.find_spec(ext) is not None
    if ext_exist:
        extensions.append(ext)
    else:
        print(f"Unable to find Sphinx extension '{ext}', {warn}.")

numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"optional", "type_without_description", "BadException"}
# Run docstring validation as part of build process
numpydoc_validation_checks = {"all", "GL01", "SA04", "RT03"}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "github_url": "https://github.com/danielk333/sorts",
    "show_prev_next": True,
    "navbar_end": ["search-field.html", "navbar-icon-links.html"],
}
# NOTE: The following is required for supporting of older sphinx toolchains.
#       The "theme-switcher" templated should be added directly to navbar_end
#       above and the following lines removed when the minimum supported
#       version of pydata_sphinx_theme is 0.9.0
# Add version switcher for versions of pydata_sphinx_theme that support it
import packaging
import pydata_sphinx_theme

if packaging.version.parse(pydata_sphinx_theme.__version__) >= packaging.version.parse("0.9.0"):
    html_theme_options["navbar_end"].insert(0, "theme-switcher")

numpydoc_show_class_members=False
html_title = "%s v%s Manual" % (project, version)
html_last_updated_fmt = '%b %d, %Y'
#html_css_files = ["numpy.css"]
html_context = {"default_mode": "dark"}

# Prevent sphinx-panels from loading bootstrap css, the pydata-sphinx-theme
# already loads it
panels_add_bootstrap_css = False

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = []  # ['_static']

# Output file base name for HTML help builder.
htmlhelp_basename = "project-templatedoc"


# -- Options for gallery extension ---------------------------------------
sphinx_gallery_conf = {
     'examples_dirs': '../../examples',   # path to your example scripts
     'gallery_dirs': 'auto_gallery',  # path where to save gallery generated examples
     'filename_pattern': '/*.py',
     'ignore_pattern': r'.*__ngl\.py',
}

# Remove matplotlib agg warnings from generated doc when using plt.show
warnings.filterwarnings("ignore", category=UserWarning,
    message='Matplotlib is currently using agg, which is a'
            ' non-GUI backend, so cannot show the figure.')

# NUMPYDOC
numpydoc_class_members_toctree = False
numpydoc_class_members_toctree = False

# -----------------------------------------------------------------------------
# Autosummary
# -----------------------------------------------------------------------------

autosummary_generate = True
autosummary_generate_overwrite = True

# -----------------------------------------------------------------------------
# Intersphinx configuration
# -----------------------------------------------------------------------------
intersphinx_mapping = {
    'numpy': ('https://docs.scipy.org/doc/numpy', None),
    'scipy': ('http://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('http://matplotlib.sourceforge.net/', None),
}

# -----------------------------------------------------------------------------
# Breathe & Doxygen
# -----------------------------------------------------------------------------
breathe_projects = dict(sorts=os.path.join("..", "build", "doxygen", "xml"))
breathe_default_project = "sorts"
breathe_default_members = ("members", "undoc-members", "protected-members")

from sphinx.ext.autosummary.generate import AutosummaryRenderer

def shortname(fullname):
    parts = fullname.split(".")
    return ".".join(parts[-2:])


def fixed_init(self, app, template_dir=None):
    AutosummaryRenderer.__old_init__(self, app, template_dir)
    self.env.filters["shortname"] = shortname


AutosummaryRenderer.__old_init__ = AutosummaryRenderer.__init__
AutosummaryRenderer.__init__ = fixed_init