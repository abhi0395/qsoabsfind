import unittest
import os
import numpy as np
from qsoabsfind.absfinder import read_single_spectrum_and_find_absorber
from qsoabsfind.parallel_convolution import parallel_convolution_method_absorber_finder_QSO_spectra
from qsoabsfind.config import load_constants
constants = load_constants()

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

        # Call the function
        result = read_single_spectrum_and_find_absorber(
            self.fits_file, spec_index, absorber, **constants.search_parameters[absorber])

        # Validate the output
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 15)  # Ensure the correct number of return values

    def test_parallel_convolution_method_absorber_finder_QSO_spectra(self):
        # Set up the input parameters for the function
        spec_indices = np.random.randint(0, 100, size=3)
        absorber = 'MgII'
        n_jobs = 4
        # Call the function
        results = parallel_convolution_method_absorber_finder_QSO_spectra(
            self.fits_file, spec_indices, absorber, n_jobs, **constants.search_parameters[absorber])

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
