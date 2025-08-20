import unittest
from qsoabsfind.constants import lines, oscillator_parameters, doublet_keys, amplitude_dict

class TestConstants(unittest.TestCase):
    def setUp(self):
        self.absorbers = doublet_keys.keys()

    def test_supported_absorbers(self):
        self.assertEqual(len(self.absorbers), 7)

    def test_absorber_parameters(self):
        for metal in self.absorbers:
            self.assertEqual(len(doublet_keys[metal]), 2) # is a doublet
            self.assertTrue(any(k.startswith(metal + '_') for k in lines.keys())) # if lines are present

    def test_parameter_lengths(self):
        self.assertEqual(len(oscillator_parameters), 2 * len(self.absorbers)) # oscillator strenght of both lines are there
        self.assertEqual(len(amplitude_dict), len(self.absorbers)) # amplitudes of both lines are there

if __name__ == '__main__':
    unittest.main()
