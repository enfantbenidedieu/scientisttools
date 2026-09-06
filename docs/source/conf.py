# utile pour le male html, mais pas utile pour readthedocs
# import os
# import sys
# sys.path.insert(0, os.path.abspath("../../../scientisttools"))

project = 'scientisttools'
copyright = '2026, Duvérier DJIFACK ZEBAZE'
author = 'Duvérier DJIFACK ZEBAZE'
release = '0.2.0.post1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",              # Markdown
    "sphinx_copybutton",        # Bouton copier
    "sphinx_design",            # Grilles, cartes, boutons, cards
    "sphinx.ext.autodoc",       # Auto-doc Python
    'sphinx.ext.mathjax',       # for math equation - latext style
    "sphinx_autodoc_typehints", # Automatically add the types
    "sphinx.ext.viewcode",      # Add links to highlighted source code
    'sphinx.ext.autosummary',   # autosummary functions and class
    "numpydoc",
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    "sphinxcontrib.email",       # for mail
    "nbsphinx"                   # to add ipynb file
]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "exclude-members": "__init__, get_metadata_routing, get_params, set_fit_request, set_output, set_params"
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

pygments_style = "sphinx" # for code style highlight (sphinx, friendly, colorful, monokai, default)
html_theme = "pydata_sphinx_theme"
html_theme_options = {
    "show_prev_next": True,
    "logo": {"text": "scientisttools"},
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "navigation_depth": 2,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/enfantbenidedieu/scientisttools",
            "icon": "fab fa-github",   # icône GitHub Font Awesome
        }
    ]
}
html_static_path = ['_static']
html_css_files = ['style.css']
html_logo = "_static/scientisttools.svg"
html_favicon = "_static/scientisttools.svg"

numfig = True
napoleon_use_param = False # 
numfig_format = {
    'code-block': 'Listing %s',
    'figure': 'Fig. %s',
    'section': 'Section',
    'table': 'Table %s',
}
email_automode = True # for mail
nbsphinx_execute = "never" # to generate PDF without execute ipynb