"""
Tests for main convolution method
"""

import unittest
import os
import numpy as np
from astropy.table import Table
from qsoabsfind.absfinder import read_single_spectrum_and_find_absorber
from qsoabsfind.parallel_convolution import parallel_convolution_search
from qsoabsfind.config import load_constants
from qsoabsfind.columndensity import total_column_density
from qsoabsfind.datamodel import QSOSpecRead
from qsoabsfind.absorberutils import return_if_absorber_can_be_detected_in_a_spectrum

class TestQSOAbsFind(unittest.TestCase):

    def setUp(self):
        # Set the file path to the data file

        #SDSS
        self.sdss_fits_file = os.path.join(os.path.dirname(__file__), '..', 'data/sdss',
                                           'qso_test_spectra.fits')
        self.sdss_constant_file = os.path.join(os.path.dirname(__file__), '..', 'data/sdss',            'sdss_constants.py')

        # Ensure the file exists
        self.assertTrue(os.path.exists(self.sdss_fits_file), f"File {self.sdss_fits_file} does not  exist")
        self.assertTrue(os.path.exists(self.sdss_constant_file), f"File {self.sdss_constant_file} does not  exist")

        #DESI
        self.desi_fits_file = os.path.join(os.path.dirname(__file__), '..', 'data/desi',          'qso_test_spectra.fits')
        self.desi_constant_file = os.path.join(os.path.dirname(__file__), '..', 'data/desi',            'desi_constants.py')

        # Ensure the file exists
        self.assertTrue(os.path.exists(self.desi_fits_file), f"File {self.desi_fits_file} does not  exist")
        self.assertTrue(os.path.exists(self.desi_constant_file), f"File {self.desi_constant_file} does not  exist")

        self.sdss_constants = load_constants(constants_file=self.sdss_constant_file)
        self.desi_constants = load_constants(constants_file=self.desi_constant_file)

        self.abs_cat = Table(
            {
                "MGII_2796_EW": [0.5],
                "MGII_2803_EW": [0.35],
                "MGII_2796_EW_ERROR": [0.1],
                "MGII_2803_EW_ERROR": [0.1],
            }
        )

    def test_available_wavelength_pixels(self):
        spec_index = np.random.randint(100)
        spec = QSOSpecRead(self.sdss_fits_file, autoload=True, index = spec_index)
        kwargs = {'verbose':False, "lam_edge_sep":25, 'start_rest_wave':None, 'end_rest_wave':None, 'dv':5000}
        is_available = return_if_absorber_can_be_detected_in_a_spectrum(spec, "MgII", **kwargs)
        self.assertIn(is_available, [0,1])

    def test_convolution_method_absorber_finder_in_QSO_spectra(self):
        # Set up the input parameters for the function
        spec_index = np.random.randint(100)
        absorber="MgII"
        # Call the function
        sdss_result = read_single_spectrum_and_find_absorber(
            self.sdss_fits_file, spec_index, absorber, **self.sdss_constants.search_parameters)

        desi_absorber="CIV"
        desi_result = read_single_spectrum_and_find_absorber(
            self.desi_fits_file, spec_index, desi_absorber, **self.desi_constants.search_parameters)

        desi_result = read_single_spectrum_and_find_absorber(
            self.desi_fits_file, spec_index, desi_absorber, **self.desi_constants.search_parameters)

        # Validate the output
        self.assertIsInstance(sdss_result, dict)
        self.assertEqual(len(sdss_result), 16)  # Ensure the correct number of keys
        self.assertIn('z_abs', sdss_result)

        self.assertIsInstance(desi_result, dict)
        self.assertEqual(len(desi_result), 16)  # Ensure the correct number of keys
        self.assertIn('z_abs', desi_result)

    def test_parallel_convolution_method_absorber_finder_QSO_spectra(self):
        # Set up the input parameters for the function
        spec_indices = np.random.randint(0, 100, size=4)
        absorber = 'MgII'
        n_jobs = 6
        # Call the function
        sdss_results = parallel_convolution_search(
            self.sdss_fits_file, spec_indices, absorber, n_jobs, **self.sdss_constants.search_parameters)

        desi_absorber='CIV'
        desi_results = parallel_convolution_search(
            self.desi_fits_file, spec_indices, desi_absorber, n_jobs, **self.desi_constants.search_parameters)

        # Validate the output
        self.assertIsInstance(sdss_results, dict)
        self.assertIn('index_spec', sdss_results)
        self.assertIn('z_abs', sdss_results)

        self.assertIsInstance(desi_results, dict)
        self.assertIn('index_spec', desi_results)
        self.assertIn('z_abs', desi_results)

        if len(sdss_results['index_spec']) == 0:
            self.skipTest("Skipping test: no SDSS absorbers detected")

        if len(desi_results['index_spec']) == 0:
            self.skipTest("Skipping test: no DESI absorbers detected")

        # Continue with assertions only if absorbers exist
        self.assertGreater(len(sdss_results['index_spec']), 0)
        self.assertGreater(len(desi_results['index_spec']), 0)

        # checking if AODM column density part passes
        if len(sdss_results['index_spec']) > 0:
            spec = QSOSpecRead(self.sdss_fits_file, autoload=True, index = sdss_results['index_spec'][0])
            F_lambda = spec.flux
            error = spec.error
            wavelength = spec.wavelength
            f1, f2 = 0.6123, 0.3054
            self.abs_cat["Z_ABS"] = [sdss_results['z_abs'][0]]
            lambda1, lambda2 = ("MGII_2796", 2796.35), ("MGII_2803", 2803.52)
            Ncol = total_column_density(F_lambda, error, wavelength, self.abs_cat, f1, f2, lambda1, lambda2,continuum_error_frac=self.sdss_constants.search_parameters["continuum_error_frac"], velocity_range=300, logwave=self.sdss_constants.search_parameters["logwave"])
            self.assertEqual(len(Ncol.dtype.names), 4)
        else:
            self.skipTest("Skipping column density test: no SDSS absorbers detected")

if __name__ == '__main__':
    unittest.main()
