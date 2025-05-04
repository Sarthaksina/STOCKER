"""
Sphinx configuration for STOCKER Pro documentation.
"""
import os
import sys
from datetime import datetime

# Add the project root directory to the path
sys.path.insert(0, os.path.abspath('..'))

# Project information
project = 'STOCKER Pro'
copyright = f'{datetime.now().year}, STOCKER Pro Team'
author = 'STOCKER Pro Team'
version = '0.1.0'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',  # Automatically include docstrings
    'sphinx.ext.viewcode',  # Add links to source code
    'sphinx.ext.napoleon',  # Support for NumPy and Google style docstrings
    'sphinx.ext.intersphinx',  # Link to other projects' documentation
    'sphinx.ext.autosummary',  # Generate summaries automatically
    'sphinx.ext.coverage',  # Check documentation coverage
    'sphinx.ext.todo',  # Support for TODO items
    'sphinx.ext.imgmath',  # Math support
    'sphinx_rtd_theme',  # Read the Docs theme
    'sphinx_autodoc_typehints',  # Include type hints in the documentation
]

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
    'exclude-members': '__weakref__',
}

# Napoleon settings for Google-style docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None

# Add any paths that contain templates here, relative to this directory
templates_path = ['_templates']

# Include Python in syntax highlighting
highlight_language = 'python3'

# List of patterns to exclude from automatic documentation
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The theme to use for HTML and HTML Help pages
html_theme = 'sphinx_rtd_theme'

# Theme options
html_theme_options = {
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# The name of an image file to place at the top of the sidebar
# html_logo = '_static/logo.png'

# The name of an image file to use as favicon
# html_favicon = '_static/favicon.ico'

# Add any paths that contain custom static files
html_static_path = ['_static']

# Links to other project's documentation
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
    'tensorflow': ('https://www.tensorflow.org/api_docs/python', None),
}

# Output file base name for HTML help builder
htmlhelp_basename = 'STOCKERProdoc'

# Set todo to emit warnings
todo_emit_warnings = True

# Automatically extract typehints when specified and place them in
# descriptions of the relevant function/method
autodoc_typehints = 'description'
autoclass_content = 'both'

# Sort members by type
autodoc_member_order = 'groupwise'

# Default to non-full signature
autodoc_default_flags = ['members']

# Make sure the target is unique
autosectionlabel_prefix_document = True 