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


def load_yaml_config(yaml_file):
    """
    Load CLI argument defaults from a YAML configuration file.

    YAML keys may use either hyphens (``zabs-known-file``) or underscores
    (``zabs_known_file``) -- both forms are normalised to underscores so they
    match the argparse ``dest`` names.  Values in the file are used as
    defaults; any argument explicitly passed on the command line will
    override them.

    Args:
        yaml_file (str): Path to the YAML configuration file.

    Returns:
        dict: Mapping of argument name -> value, with ``null`` entries
        removed so they do not shadow argparse-level defaults.

    Raises:
        ImportError: If PyYAML is not installed.
        FileNotFoundError: If *yaml_file* does not exist.
        ValueError: If the file does not contain a YAML mapping.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required to use --config. "
            "Install it with: pip install pyyaml"
        )

    if not os.path.isfile(yaml_file):
        raise FileNotFoundError(f"Config file not found: {yaml_file}")

    with open(yaml_file, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    if not isinstance(config, dict):
        raise ValueError(
            f"Config file must be a YAML mapping (key: value pairs): {yaml_file}"
        )

    # Normalise hyphenated keys to underscores so they match argparse dest names,
    # then drop null values so they do not override argparse-level defaults.
    return {k.replace('-', '_'): v for k, v in config.items() if v is not None}