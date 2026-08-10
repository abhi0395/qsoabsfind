"""
This module contains data model classes for reading and handling QSO spectra
and absorber catalog FITS files.
"""
import time
import os
import logging
from .io import read_fits_file, read_any_fits_file
from .utils import get_all_extnames

logger = logging.getLogger(__name__)

class QSOSpecRead:
    """
    A class to read and handle QSO spectra from a FITS file containing FLUX, ERROR, WAVELENGTH, and METADATA extensions.
    """

    def __init__(self, fits_file, index=None, autoload=False, verbose=True):
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
        self.header = None
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
        if not os.path.exists(self.fits_file):
            raise IOError(f"ERROR: {self.fits_file} does not exist")
        start_time = time.time()
        self.header, self.flux, self.error, self.wavelength, self.metadata = read_fits_file(self.fits_file, self.index)
        if self.verbose:
            logger.info("Time taken to read %s: %.2f seconds", self.fits_file, time.time() - start_time)

    def get_metadata(self, asdict=False):
        """
        Returns the METADATA data with keyword handling (must be used after read_fits() option).

        Args:
            asdict (bool): if True, metadata will be returned as a dictionary, otherwise astropy.table

        Returns:
            dict or Table: The metadata data with keywords (if asdict=True), otherwise a Table
        """
        if self.metadata is None:
            raise ValueError("ERROR: there is no metadata available, use read_fits()")
        elif asdict:
            details_dict = {key: self.metadata[key] for key in self.metadata.dtype.names}
            return details_dict
        else:
            return self.metadata

class AbsorberData():
    """
    A class to read and handle Absorber table FITS file
    containing ABSORBER, METADATA, and optionally COLUMN_DENSITY extensions.
    """

    def __init__(self, filepath, autoload=False, verbose=True):
        """
        Initializes the AbsorberData class.

        Args:
            filepath (str): Path to the FITS file containing Absorber catalog.
            autoload (bool): if True, class itself will load the data (default=False),
            verbose (bool): if want to print time info
        """
        self.filepath = filepath
        self.verbose = verbose
        self.autoload=True

        # Get all extension names
        self.extnames = get_all_extnames(self.filepath)

        if self.autoload:
            self.read_catalog()

    def read_catalog(self):
        """Read catalog from the file"""

        start_time = time.time()
        # Temporary dictionary to store all data from different extensions
        all_data = {}

        # Read each extension
        for idx, ext_name, hdu_type in self.extnames:
            if self.verbose:
                logger.info("Reading extension '%s' (%s)", ext_name, hdu_type)

            # Use index for PRIMARY HDU, name for others
            hdu_identifier = idx if ext_name in ['PRIMARY', f'HDU_{idx}'] else ext_name
            hdr, data = read_any_fits_file(self.filepath, hdu_name=hdu_identifier)
            all_data[ext_name] = data

        # Assign specific extensions as attributes
        self.catalog = all_data.get('ABSORBER', None)
        self.metadata = all_data.get('METADATA', None)
        self.column_density = all_data.get('COLUMN_DENSITY', None)  # May be None
        self.qso_info = all_data.get('QSO_INFO', None)  # May be None


        # Store header from last read
        self.header = hdr

        # Check required extensions
        if self.catalog is None:
            raise ValueError(f"Required extension 'ABSORBER' not found in {self.filepath}")
        if self.metadata is None:
            raise ValueError(f"Required extension 'METADATA' not found in {self.filepath}")

        if self.verbose:
            logger.info("Loaded %d extensions from %s", len(self.extnames), self.filepath)
            logger.info("  ABSORBER: %d rows", len(self.catalog))
            logger.info("  METADATA: %d rows", len(self.metadata))
            if self.column_density is not None:
                logger.info("  COLUMN_DENSITY: %d rows", len(self.column_density))
            else:
                logger.info("  COLUMN_DENSITY: Not present (optional)")
            if self.qso_info is not None:
                logger.info("  QSO_INFO: %d rows", len(self.qso_info))
            else:
                logger.info("  QSO_INFO: Not present (optional)")
            logger.info("Time taken to read catalog from %s: %.2f seconds", self.filepath, time.time() - start_time)
