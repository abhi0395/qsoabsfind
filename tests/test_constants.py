"""
Tests for constants
"""
import unittest
from qsoabsfind.constants import (
    lines, oscillator_parameters, doublet_keys, amplitude_dict,
    FIT_PARAM_SNR, GAUSS_FIT_BOOT_ITER_FACTOR, GAUSS_FIT_BOOT_WARM_SPREAD,
)

class TestConstants(unittest.TestCase):
    def setUp(self):
        self.absorbers = doublet_keys.keys()

    def test_supported_absorbers(self):
        self.assertEqual(len(self.absorbers), 9)

    def test_absorber_parameters(self):
        for metal in self.absorbers:
            self.assertEqual(len(doublet_keys[metal]), 2) # is a doublet
            self.assertTrue(any(k.startswith(metal + '_') for k in lines.keys())) # if lines are present

    def test_parameter_lengths(self):
        self.assertEqual(len(oscillator_parameters), 2 * len(self.absorbers)) # oscillator strenght of both lines are there
        self.assertEqual(len(amplitude_dict), len(self.absorbers)) # amplitudes of both lines are there


class TestBootstrapAndFitParamConstants(unittest.TestCase):

    def test_fit_param_snr_positive(self):
        self.assertGreater(FIT_PARAM_SNR, 0)

    def test_boot_iter_factor_between_zero_and_one(self):
        self.assertGreater(GAUSS_FIT_BOOT_ITER_FACTOR, 0)
        self.assertLessEqual(GAUSS_FIT_BOOT_ITER_FACTOR, 1)

    def test_boot_warm_spread_between_zero_and_one(self):
        self.assertGreater(GAUSS_FIT_BOOT_WARM_SPREAD, 0)
        self.assertLess(GAUSS_FIT_BOOT_WARM_SPREAD, 1)

if __name__ == '__main__':
    unittest.main()
