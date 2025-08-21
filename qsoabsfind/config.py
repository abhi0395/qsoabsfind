"""
This function loads parameters from the user-provided parameter file.
"""
import os
import importlib.util

def load_constants(constants_file):
    """
    Load constants from a user-provided constant file.
    Args:
        constants_file (str): constant filename

    Returns:
    -------
    module
        The module (either user-defined or default) containing the constants.
    """

    if constants_file is None:
        raise ValueError("ERROR: Provide constant file with search parameters")

    # Check if the environment variable for the constants file is set
    if constants_file and os.path.isfile(constants_file):
        spec = importlib.util.spec_from_file_location("user_constants", constants_file)
        user_constants = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(user_constants)
        return user_constants
    else:
        raise ValueError("ERROR: Input constant file does not exist..")