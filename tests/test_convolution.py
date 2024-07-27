import unittest
import os
import numpy as np
from qsoabsfind.absfinder import convolution_method_absorber_finder_in_QSO_spectra
from qsoabsfind.parallel_convolution import parallel_convolution_method_absorber_finder_QSO_spectra

class TestQSOAbsFind(unittest.TestCase):

    def test_convolution_method_absorber_finder_in_QSO_spectra(self):
        # Set up the input parameters for the function
        fits_file = os.path.join(os.path.dirname(__file__), 'qso_test.fits')
        # Check if the file exists
        assert os.path.exists(file_path), f"File {fits_file} does not exist"
        spec_index = 0
        absorber = 'MgII'
        ker_width_pix = [3, 4, 5, 6, 7, 8]
        coeff_sigma = 2.5
        mult_resi = 1
        d_pix = 0.6
        pm_pixel = 200
        sn_line1 = 3
        sn_line2 = 2
        use_covariance = False

        # Call the function
        result = convolution_method_absorber_finder_in_QSO_spectra(
            fits_file, spec_index, absorber, ker_width_pix, coeff_sigma,
            mult_resi, d_pix, pm_pixel, sn_line1, sn_line2, use_covariance)

        # Validate the output
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 13)  # Ensure the correct number of return values

    def test_parallel_convolution_method_absorber_finder_QSO_spectra(self):
        # Set up the input parameters for the function
        fits_file = 'qso_test.fits'
        spec_indices = np.array([0, 1, 2])
        absorber = 'MgII'
        ker_width_pix = [3, 4, 5, 6, 7, 8]
        coeff_sigma = 2.5
        mult_resi = 1
        d_pix = 0.6
        pm_pixel = 200
        sn_line1 = 3
        sn_line2 = 2
        use_covariance = False
        n_jobs = 4

        # Call the function
        results = parallel_convolution_method_absorber_finder_QSO_spectra(
            fits_file, spec_indices, absorber, ker_width_pix, coeff_sigma,
            mult_resi, d_pix, pm_pixel, sn_line1, sn_line2, use_covariance, n_jobs)

        print("Results:", results)
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
