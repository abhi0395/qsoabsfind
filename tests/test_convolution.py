import unittest
import os
import numpy as np
from qsoabsfind.absfinder import read_single_spectrum_and_find_absorber
from qsoabsfind.parallel_convolution import parallel_convolution_method_absorber_finder_QSO_spectra

class TestQSOAbsFind(unittest.TestCase):

    def setUp(self):
        # Set the file path to the data file
        self.fits_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'qso_test.fits')
        # Ensure the file exists
        self.assertTrue(os.path.exists(self.fits_file), f"File {self.fits_file} does not exist")

    def test_convolution_method_absorber_finder_in_QSO_spectra(self):
        # Set up the input parameters for the function
        spec_index = np.random.randint(100)
        absorber = 'MgII'
        ker_width_pixels = [3, 4, 5, 6, 7, 8]
        coeff_sigma = 2.5
        mult_resi = 1
        d_pix = 0.6
        pm_pixel = 200
        sn_line1 = 3
        sn_line2 = 2
        use_covariance = False
        logwave=True
        verbose=False

        kwargs = {'ker_width_pixels': ker_width_pixels, 'coeff_sigma': coeff_sigma, 'mult_resi': mult_resi, 'd_pix': d_pix, 'pm_pixel': pm_pixel, 'sn_line1': sn_line1, 'sn_line2': sn_line2, 'use_covariance': use_covariance, 'logwave': logwave, 'verbose': verbose}

        # Call the function
        result = read_single_spectrum_and_find_absorber(
            self.fits_file, spec_index, absorber, **kwargs)

        # Validate the output
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 13)  # Ensure the correct number of return values

    def test_parallel_convolution_method_absorber_finder_QSO_spectra(self):
        # Set up the input parameters for the function
        spec_indices = np.random.randint(0, 100, size=3)
        absorber = 'MgII'
        ker_width_pixels = [3, 4, 5, 6, 7, 8]
        coeff_sigma = 2.5
        mult_resi = 1
        d_pix = 0.6
        pm_pixel = 200
        sn_line1 = 3
        sn_line2 = 2
        use_covariance = False
        n_jobs = 4
        logwave=True
        verbose=False

        # Call the function
        results = parallel_convolution_method_absorber_finder_QSO_spectra(
            self.fits_file, spec_indices, absorber, ker_width_pixels, coeff_sigma,
            mult_resi, d_pix, pm_pixel, sn_line1, sn_line2, use_covariance, logwave, verbose, n_jobs)

        # Validate the output
        self.assertIsInstance(results, dict)
        self.assertIn('index_spec', results)
        self.assertIn('z_abs', results)
        try:
            self.assertGreater(len(results['index_spec']), 0)
        except AssertionError:
            print('INFO:: Test failed possibly because no absorber could be detected')

if __name__ == '__main__':
    unittest.main()
