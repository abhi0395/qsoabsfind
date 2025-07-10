# config.py
"""This module handles dynamic loading of constants for the QSO absorber
finder. It first loads the default constants from the constants module and then
optionally overwrites them with user-provided constants from a specified file.

If an environment variable 'QSO_CONSTANTS_FILE' is set and points to a valid file, the constants
from that file will be loaded and used instead of the default ones.

Usage:
1. Ensure the default constants are defined in the constants module.
2. Optionally provide a custom constants file path in the environment variable 'QSO_CONSTANTS_FILE'.
3. Import constants from this module in other parts of the application to use the loaded constants.
"""

import os
import importlib.util
import qsoabsfind.constants as default_constants

def load_constants(constants_file=None, verbose=False):
    """
    Load constants used by qsoabsfind.

    This function checks the QSO_CONSTANTS_FILE environment variable.
    If it points to a valid Python file, that file is dynamically loaded and used
    as the constants module. If the variable is unset or invalid, it falls back
    to the built-in constants in `qsoabsfind.constants`.

    Args:
        constants_file (str): constant filename (optional)
        verbose (bool): if want to print statements

    Returns:
    -------
    module
        The module (either user-defined or default) containing the constants.
    """
    if constants_file is None:
        constants_file = os.environ.get('QSO_CONSTANTS_FILE')

    # Check if the environment variable for the constants file is set
    if constants_file and os.path.isfile(constants_file):
        try:
            spec = importlib.util.spec_from_file_location("user_constants", constants_file)
            user_constants = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(user_constants)
            return user_constants
        except Exception as e:
            print(f"Error loading user constants: {e}. Falling back to default constants.")
            return default_constants
    else:
        if verbose:
            print("Using default constants.")
        return default_constants
