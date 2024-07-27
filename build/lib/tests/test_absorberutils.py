import unittest
import numpy as np
from qsoabsfind.absorberutils import (
    estimate_local_sigma_conv_array, group_and_weighted_mean_selection_function,
    redshift_estimate
)
from qsoabsfind.absfinder import find_valid_indices

class TestAbsorberUtils(unittest.TestCase):
    def setUp(self, nzabs=10, nwave=4500):
        self.conv_array = np.random.random(nwave)
        self.pm_pixel = 200
        self.master_list_of_pot_absorber = np.random.random(nzabs)
        self.residual = np.random.random(nwave)
        self.nzabs =nzabs
        self.nwave=nwave

    def test_estimate_local_sigma_conv_array(self):
        sigma_cr = estimate_local_sigma_conv_array(self.conv_array, self.pm_pixel)
        self.assertEqual(len(sigma_cr), len(self.conv_array))

    def test_group_and_weighted_mean_selection_function(self):
        grouped_z = group_and_weighted_mean_selection_function(self.master_list_of_pot_absorber, self.residual)
        self.assertGreater(len(grouped_z), 0)

    def test_find_valid_indices(self):
        our_z = np.random.random(self.nzabs)
        residual_our_z = np.random.random(self.nzabs)
        lam_search = np.random.random(self.nwave)
        conv_arr = np.random.random(self.nwave)
        sigma_cr = np.random.random(self.nwave)
        coeff_sigma = 2.5
        d_pix = 0.6
        beta = 0.5
        line1 = 2796.35
        line2 = 2803.52
        new_our_z, new_res_arr = find_valid_indices(our_z, residual_our_z, lam_search, conv_arr, sigma_cr, coeff_sigma, d_pix, beta, line1, line2)
        if lam_search.size>0:
            self.assertGreater(len(new_our_z), 0)
        else:
            print('INFO:: test passed, just that QSO redshift is smaller than minimum redshift')

    def test_redshift_estimate(self):
        gauss_params = np.array([1, 2796.35, 1, 1, 2803.52, 1])
        std_params = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        line1 = 2796.35
        line2 = 2803.52
        z_corr, z_err = redshift_estimate(gauss_params, std_params, line1, line2)
        self.assertIsNotNone(z_corr)
        self.assertIsNotNone(z_err)

if __name__ == '__main__':
    unittest.main()
