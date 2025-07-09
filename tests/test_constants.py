import unittest
from qsoabsfind.config import load_constants
constants = load_constants()

lines, search_parameters, speed_of_light = constants.lines, constants.search_parameters, constants.speed_of_light
doublet_keys = constants.doublet_keys
oscillator_strengths = constants.oscillator_parameters
amplitude_dict = constants.amplitude_dict

class TestConstants(unittest.TestCase):
    def test_line_data(self):
        self.assertIn('MgII_2796', lines)
        self.assertIn('CIV_1548', lines)

    def test_search_parameters(self):
        self.assertIn('MgII', search_parameters)
        self.assertIn('CIV', search_parameters)

    def test_speed_of_light(self):
        self.assertEqual(speed_of_light, 3e5)

    def test_supported_absorber(self):
        self.assertEqual(len(doublet_keys.keys()), 7)

    def test_parameter_lengths(self):
        self.assertEqual(len(oscillator_strengths.keys()), 2 * len(doublet_keys.keys()), 2 * len(amplitude_dict))


if __name__ == '__main__':
    unittest.main()
