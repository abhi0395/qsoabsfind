import unittest
from qsoabsfind.config import lines, search_parameters, speed_of_light

class TestConstants(unittest.TestCase):
    def test_line_data(self):
        self.assertIn('MgII_2796', lines)
        self.assertIn('CIV_1548', lines)

    def test_search_parameters(self):
        self.assertIn('MgII', search_parameters)
        self.assertIn('CIV', search_parameters)

    def test_speed_of_light(self):
        self.assertEqual(speed_of_light, 3e5)

if __name__ == '__main__':
    unittest.main()
