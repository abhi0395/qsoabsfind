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

    def test_absorber_parameters(self):
        for metal in self.absorbers:
            self.assertTrue(any(k.startswith(metal + '_') for k in lines.keys()))
            self.assertIn(metal, search_parameters)

    def test_speed_of_light(self):
        self.assertEqual(speed_of_light, 3e5)

    def test_parameter_lengths(self):
        self.assertEqual(len(oscillator_strengths), 2 * len(self.absorbers))
        self.assertEqual(len(amplitude_dict), len(self.absorbers))


if __name__ == '__main__':
    unittest.main()
