import unittest
import numpy as np
from qsoabsfind.parallel_convolution import parallel_convolution_method_absorber_finder_QSO_spectra

class TestParallelConvolutionMethodSingleSpectrum(unittest.TestCase):
    def setUp(self):
        self.fits_file = "qso_test.fits"
        self.spec_indices = np.arange(10)
        self.absorber = "MgII"
        self.ker_width_pix = [3, 4, 5]
        self.coeff_sigma = 2.5
        self.mult_resi = 1.0
        self.d_pix = 0.6
        self.pm_pixel = 200
        self.sn_line1 = 3
        self.sn_line2 = 2
        self.use_covariance = False
        self.n_jobs = 4

    def test_parallel_convolution_method(self):
        results = parallel_convolution_method_absorber_finder_QSO_spectra(
            self.fits_file, self.spec_indices, self.absorber, self.ker_width_pix,
            self.coeff_sigma, self.mult_resi, self.d_pix, self.pm_pixel,
            self.sn_line1, self.sn_line2, self.use_covariance, self.n_jobs
        )
        self.assertIsNotNone(results)
        self.assertGreater(len(results['index_spec']), 0)

if __name__ == '__main__':
    unittest.main()
