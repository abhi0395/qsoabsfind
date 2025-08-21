"""
Tests for utils.py
"""

import unittest
import numpy as np
from qsoabsfind.utils import (convolution_fun,
                              compute_doublet_amplitudes
)

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

    def test_apmplitude_def(self):
        A1, A2 = compute_doublet_amplitudes(0.5, 0.5, 0.7)
        self.assertTrue(A1<=1)
        self.assertTrue(A2<=1)

if __name__ == '__main__':
    unittest.main()
