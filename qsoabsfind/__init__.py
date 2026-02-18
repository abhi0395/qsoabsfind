"""Qsoabsfind module initialization."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("qsoabsfind")
except PackageNotFoundError:
    __version__ = "unknown"