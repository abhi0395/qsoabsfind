"""Qsoabsfind module initialization."""
import logging
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("qsoabsfind")
except PackageNotFoundError:
    __version__ = "unknown"

# Standard practice for library packages: prevent "No handlers found" warnings
# when the caller has not configured logging.
logging.getLogger(__name__).addHandler(logging.NullHandler())