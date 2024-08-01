import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import qsoabsfind
from datetime import datetime
import subprocess

project = 'qsoabsfind'
author = 'Abhijeet Anand'
release = '0.1.0'
copyright = f'2021-{datetime.now().year}, Abhijeet Anand'

html_context = {
    'current_year': datetime.now().year,
    "display_github": True, # Add 'Edit on Github' link instead of 'View page source'
    "github_user": "abhi0395",  # Username
    "github_repo": "qsoabsfind",  # Repo name
    "github_version": "main",  # Version
    "conf_py_path": "/docs/index.rst",  # Path in the checkout to the docs root
    "last_updated": datetime.now(),
    "commit": False,
}

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_autodoc_typehints',
]

html_theme = 'sphinx_rtd_theme'

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
}

apoleon_google_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
