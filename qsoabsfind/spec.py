"""
This script contains a classes and functions to read a given spectra fits files.
"""

from .io import read_fits_file
from .utils import elapsed
import time

class QSOSpecRead:
    """
    A class to read and handle QSO spectra from a FITS file containing FLUX, ERROR, WAVELENGTH, and METADATA extensions."""

    def __init__(self, fits_file, index=None):
        """
        Initializes the QSOSpecRead class.

        Args:
            fits_file (str): Path to the FITS file containing QSO spectra.
            index (int, list, or np.ndarray, optional): Index or indices of the rows to load. Default is None.
        """
        self.fits_file = fits_file
        self.flux = None
        self.error = None
        self.wavelength = None
        self.metadata = None
        self.index = index
        self.read_fits()

    def read_fits(self):
        """
        Reads the FITS file and measures the time taken for the
        operation.
        """
        start_time = time.time()
        self.flux, self.error, self.wavelength, self.metadata = read_fits_file(self.fits_file, self.index)
        elapsed(start_time, "\nTime taken to read FITS file")

    def get_metadata(self):
        """
        Returns the METADATA data with keyword handling.

        Returns:
            dict: The metadata data with keywords.
        """
        details_dict = {key: self.metadata[key] for key in self.metadata.dtype.names}
        return details_dict
