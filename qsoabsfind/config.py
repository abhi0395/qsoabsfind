# config.py

"""
This module handles dynamic loading of constants for the QSO absorber finder.
It first loads the default constants from the constants module and then optionally
overwrites them with user-provided constants from a specified file.

If an environment variable 'QSO_CONSTANTS_FILE' is set and points to a valid file, the constants
from that file will be loaded and used instead of the default ones.

Usage:
1. Ensure the default constants are defined in the constants module.
2. Optionally provide a custom constants file path in the environment variable 'QSO_CONSTANTS_FILE'.
3. Import constants from this module in other parts of the application to use the loaded constants.
"""

import importlib.util
import os

from .constants import *

def load_constants(constants_file):
    """
    Dynamically loads constants from a user-provided file.

    Parameters:
    constants_file (str): Path to the file containing user-defined constants.
    """
    spec = importlib.util.spec_from_file_location("user_constants", constants_file)
    user_constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(user_constants)

    # Update the globals dictionary with the user-defined constants
    globals().update(user_constants.__dict__)

# Check if the environment variable for the constants file is set
constants_file = os.getenv('QSO_CONSTANTS_FILE')
if constants_file and os.path.isfile(constants_file):
    load_constants(constants_file)
