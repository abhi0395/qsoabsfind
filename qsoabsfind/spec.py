"""
This script contains a class and functions to read a given spectra fits file.
"""

from .io import read_fits_file
from .utils import elapsed
import time

class QSOSpecRead:
    """
    A class to read and handle QSO spectra from a FITS file containing FLUX, ERROR, WAVELENGTH, and METADATA extensions."""

    def __init__(self, fits_file, index=None, autoload=False, verbose=False):
        """
        Initializes the QSOSpecRead class.

        Args:
            fits_file (str): Path to the FITS file containing QSO spectra.
            index (int, list, or np.ndarray, optional): Index or indices of the rows to load. Default is None.
            autoload (bool): if True, class itself will load the data (default=False), 
                             in True case, user does not need to use available class functions.
            verbose (bool): if want to print time info
        """
        self.fits_file = fits_file
        self.flux = None
        self.error = None
        self.wavelength = None
        self.metadata = None
        self.index = index
        self.verbose = verbose
        self.autoload = autoload
        if self.autoload:
            self.read_fits()

    def read_fits(self):
        """
        Reads the FITS file and measures the time taken for the operation.
        """
        start_time = time.time()
        self.flux, self.error, self.wavelength, self.metadata = read_fits_file(self.fits_file, self.index)
        if self.verbose:
            elapsed(start_time, "\nINFO: Time taken to read {self.fits_file}")

    def get_metadata(self, asdict=False):
        """
        Returns the METADATA data with keyword handling (must be used after read_fits() option).
        
        Args:
            asdict (bool): if True, metadata will be returned as a dictionary, otherwise astropy.table
            
        Returns:
            dict or Table: The metadata data with keywords (if asdict=True), otherwise a Table
        """
        if self.metadata is None:
            raise ValueError(f"ERROR: there is no metadata available, use read_fits()")
        if asdict:
            details_dict = {key: self.metadata[key] for key in self.metadata.dtype.names}
            return details_dict
        else:
            return self.metadata
            
