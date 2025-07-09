import unittest
import os
import numpy as np
from astropy.table import Table
from qsoabsfind.absfinder import read_single_spectrum_and_find_absorber
from qsoabsfind.parallel_convolution import parallel_convolution_method_absorber_finder_QSO_spectra
from qsoabsfind.config import load_constants
from qsoabsfind.columndensity import total_column_density
from qsoabsfind.spec import QSOSpecRead

constants = load_constants()

class TestQSOAbsFind(unittest.TestCase):

    def setUp(self):
        # Set the file path to the data file
        #SDSS
        self.sdss_fits_file = os.path.join(os.path.dirname(__file__), '..', 'data/sdss', 'qso_test_spectra.fits')
        # Ensure the file exists
        self.assertTrue(os.path.exists(self.sdss_fits_file), f"File {self.sdss_fits_file} does not exist")
        #DESI
        self.desi_fits_file = os.path.join(os.path.dirname(__file__), '..', 'data/desi',            'qso_test_spectra.fits')
        # Ensure the file exists
        self.assertTrue(os.path.exists(self.desi_fits_file), f"File {self.desi_fits_file} does not  exist")

    def test_convolution_method_absorber_finder_in_QSO_spectra(self):
        # Set up the input parameters for the function
        spec_index = np.random.randint(500)
        absorber="MgII"
        # Call the function
        sdss_result = read_single_spectrum_and_find_absorber(
            self.sdss_fits_file, spec_index, absorber, **constants.search_parameters[absorber])

        desi_absorber="CIV"
        constants.search_parameters[desi_absorber]["logwave"]=False
        desi_result = read_single_spectrum_and_find_absorber(
            self.desi_fits_file, spec_index, desi_absorber, **constants.search_parameters[desi_absorber])

        # Validate the output
        self.assertIsInstance(sdss_result, tuple)
        self.assertEqual(len(sdss_result), 15)  # Ensure the correct number of return values

        self.assertIsInstance(desi_result, tuple)
        self.assertEqual(len(desi_result), 15)  # Ensure the correct number of return values

    def test_parallel_convolution_method_absorber_finder_QSO_spectra(self):
        # Set up the input parameters for the function
        spec_indices = np.random.randint(0, 500, size=3)
        absorber = 'MgII'
        n_jobs = 4
        # Call the function
        sdss_results = parallel_convolution_method_absorber_finder_QSO_spectra(
            self.sdss_fits_file, spec_indices, absorber, n_jobs, **constants.search_parameters[absorber])

        desi_absorber='CIV'
        constants.search_parameters[desi_absorber]["logwave"]=False
        desi_results = parallel_convolution_method_absorber_finder_QSO_spectra(
            self.desi_fits_file, spec_indices, desi_absorber, n_jobs, **constants.search_parameters[desi_absorber])

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
            abs_cat = Table()
            abs_cat["MGII_2796_EW"] = sdss_results["ew_1_mean"]
            abs_cat["MGII_2803_EW"] = sdss_results["ew_2_mean"]
            abs_cat["MGII_2796_EW_ERROR"] = sdss_results["ew_1_error"]
            abs_cat["MGII_2803_EW_ERROR"] = sdss_results["ew_2_error"]
            abs_cat["Z_ABS"] = sdss_results["z_abs"]
            f1, f2 = 0.6123, 0.3054
            lambda1, lambda2 = (f1, 2796.35), (f2, 2803.52)
            Ncol = total_column_density(F_lambda, error, wavelength, abs_cat, f1, f2, lambda1, lambda2, velocity_range=300)
            self.assertEqual(len(Ncol.keys), 4)

if __name__ == '__main__':
    unittest.main()
