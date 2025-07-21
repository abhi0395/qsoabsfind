import unittest
import numpy as np
from qsoabsfind.utils import convolution_fun

class TestUtils(unittest.TestCase):
    def setUp(self):
        self.absorber = "MgII"
        self.residual = np.random.random(4500)
        self.width = 3.0
        self.f1 = 0.5
        self.f2 = 0.25
        self.wave_res=0.0001
        self.index=None
        self.log=True

    def test_convolution_fun(self):
        result = convolution_fun(self.absorber, self.residual, self.width, self.log, self.wave_res, self.index, self.f1, self.f2)
        self.assertEqual(len(result), len(self.residual))

if __name__ == '__main__':
    unittest.main()
