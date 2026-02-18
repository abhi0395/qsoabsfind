import unittest
import numpy as np

from qsoabsfind.ew import return_line_centers, \
                        measure_absorber_properties_double_gaussian


class TestEW(unittest.TestCase):

    def test_return_line_centers(self):
        """Kernel name should return two valid line centers."""
        l1, l2 = return_line_centers("MgII")
        self.assertTrue(np.isfinite(l1))
        self.assertTrue(np.isfinite(l2))
        self.assertLess(l1, l2)

    def test_return_line_centers_invalid(self):
        """Invalid kernel should raise ValueError."""
        with self.assertRaises(ValueError):
            return_line_centers("INVALID_KERNEL")

    def test_measure_absorber_properties_return_length_empty(self):
        """Function should always return 11 values even for empty absorber list."""
        wavelength = np.linspace(5000, 6000, 2000)
        flux = np.ones_like(wavelength)
        error = np.full_like(wavelength, 0.05)

        result = measure_absorber_properties_double_gaussian(
            index=0,
            wavelength=wavelength,
            flux=flux,
            error=error,
            absorber_redshift=[],   # no absorbers
            bound=None,
            use_kernel="MgII",
            d_pix=np.median(np.diff(wavelength)),
            num_iter=200,
            window=5,
            use_covariance=False,
            nboot=None,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 11)

    def test_measure_absorber_properties_return_length_nonempty(self):
        """Function should return 11 values when absorbers are present."""
        z_true = 0.6
        line1, line2 = return_line_centers("MgII")

        mu1_obs = line1 * (1 + z_true)
        mu2_obs = line2 * (1 + z_true)

        wavelength = np.linspace(mu1_obs - 40, mu2_obs + 40, 2500)

        # Build simple normalized double Gaussian
        amp1, sig1 = 0.25, 0.9
        amp2, sig2 = 0.12, 0.9

        lam_rest = wavelength / (1 + z_true)
        model = 1.0 - amp1 * np.exp(-(lam_rest - line1) ** 2 / (2 * sig1 ** 2)) \
                    - amp2 * np.exp(-(lam_rest - line2) ** 2 / (2 * sig2 ** 2))

        error = np.full_like(wavelength, 0.02)
        flux = model + np.random.normal(0, error)

        result = measure_absorber_properties_double_gaussian(
            index=0,
            wavelength=wavelength,
            flux=flux,
            error=error,
            absorber_redshift=[z_true],
            bound=None,
            use_kernel="MgII",
            d_pix=np.median(np.diff(wavelength)),
            num_iter=2000,
            window=5,
            use_covariance=False,
            nboot=None,
        )

        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 11)

        z_array = result[0]
        EW1 = result[4]
        EW2 = result[5]
        EW_tot = result[6]

        self.assertEqual(z_array.shape, (1,))
        self.assertEqual(EW1.shape, (1,))
        self.assertEqual(EW2.shape, (1,))
        self.assertEqual(EW_tot.shape, (1,))

        # Total EW should equal sum of components (within tolerance)
        self.assertAlmostEqual(EW_tot[0], EW1[0] + EW2[0], places=4)


if __name__ == "__main__":
    unittest.main()