import unittest
from qsoabsfind.config import load_constants

constants = load_constants()
lines = constants.lines
search_parameters = constants.search_parameters
speed_of_light = constants.speed_of_light
doublet_keys = constants.doublet_keys
oscillator_strengths = constants.oscillator_parameters
amplitude_dict = constants.amplitude_dict

class TestConstants(unittest.TestCase):
    def setUp(self):
        self.absorbers = doublet_keys.keys()

    def test_supported_absorbers(self):
        self.assertEqual(len(self.absorbers), 7)

    def test_absorber_parameters(self):
        for metal in self.absorbers:
            self.assertEqual(len(doublet_keys[metal]), 2) # is a doublet
            self.assertTrue(any(k.startswith(metal + '_') for k in lines.keys())) # if lines are present
            self.assertIn(metal, search_parameters) # in search parameters dict, metal is present
            self.assertIn('dv' , lines)

    def test_parameter_lengths(self):
        self.assertEqual(len(oscillator_strengths), 2 * len(self.absorbers)) # oscillator strenght of both lines are there
        self.assertEqual(len(amplitude_dict), len(self.absorbers)) # amplitudes of both lines are there


if __name__ == '__main__':
    unittest.main()
