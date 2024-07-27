import unittest
import numpy as np
from qsoabsfind.utils import convolution_fun

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.absorber = "MgII"
        self.residual = np.random.random(4500)
        self.width = 3.0
        self.amp_ratio = 0.5

    def test_convolution_fun(self):
        result = convolution_fun(self.absorber, self.residual, self.width, self.amp_ratio)
        self.assertEqual(len(result), len(self.residual))

if __name__ == '__main__':
    unittest.main()
