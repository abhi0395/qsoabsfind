from .io import read_fits_file
from .utils import elapsed
import time

class QSOSpecRead:
    """
    A class to read and handle QSO spectra from a FITS file containing FLUX, ERROR, WAVELENGTH, and TGTDETAILS extensions."""

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
        self.tgtdetails = None
        self.index = index
        self.read_fits()

    def read_fits(self):
        """
        Reads the FITS file and measures the time taken for the
        operation.
        """
        start_time = time.time()
        self.flux, self.error, self.wavelength, self.tgtdetails = read_fits_file(self.fits_file, self.index)
        elapsed(start_time, "\nTime taken to read FITS file")

    def get_tgtdetails(self):
        """
        Returns the TGTDETAILS data with keyword handling.

        Returns:
            dict: The tgtdetails data with keywords.
        """
        details_dict = {key: self.tgtdetails[key] for key in self.tgtdetails.dtype.names}
        return details_dict
