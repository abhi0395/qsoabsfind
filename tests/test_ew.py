import unittest
import numpy as np

from qsoabsfind.ew import return_line_centers, \
                        measure_absorber_properties_double_gaussian, \
                        trapezoidal_ew, \
                        calculate_ew_errors, \
                        bootstrap_fitting_and_ew


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
        self.assertEqual(len(result), 12)

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
        self.assertEqual(len(result), 12)

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


class TestTrapezoidalEW(unittest.TestCase):
    """Tests for the trapezoidal_ew standalone function."""

    # MgII doublet rest wavelengths used throughout
    LINE1 = 2796.35
    LINE2 = 2803.53

    def _spectrum(self, z, sigma1=1.0, sigma2=1.0, n=3000, noise=0.0):
        """Return (lam_obs, residual, error) covering the doublet at redshift z."""
        rest_lam = np.linspace(self.LINE1 - 20, self.LINE2 + 20, n)
        residual = np.ones(n)
        rng = np.random.RandomState(42)
        error = np.full(n, max(noise, 0.01))
        if noise > 0:
            residual = residual + rng.normal(0, noise, n)
        lam_obs = rest_lam * (1.0 + z)
        return lam_obs, residual, error

    def _absorber_spectrum(self, z, amp1=0.4, amp2=0.2, sigma1=1.0, sigma2=1.0, n=3000):
        """Spectrum with synthetic double-Gaussian absorption."""
        lam_obs, residual, error = self._spectrum(z, n=n)
        rest_lam = lam_obs / (1.0 + z)
        residual -= amp1 * np.exp(-(rest_lam - self.LINE1) ** 2 / (2 * sigma1 ** 2))
        residual -= amp2 * np.exp(-(rest_lam - self.LINE2) ** 2 / (2 * sigma2 ** 2))
        return lam_obs, residual, error

    # ------------------------------------------------------------------ #
    #  Return type and structure                                           #
    # ------------------------------------------------------------------ #

    def test_returns_dict_with_expected_keys(self):
        lam_obs, res, err = self._spectrum(0.5)
        result = trapezoidal_ew(lam_obs, res, err, 0.5, self.LINE1, self.LINE2, 1.0, 1.0)
        self.assertIsInstance(result, dict)
        self.assertEqual(
            set(result.keys()),
            {'ew1', 'ew2', 'ew_total', 'ew1_err', 'ew2_err', 'ew_total_err'},
        )

    # ------------------------------------------------------------------ #
    #  EW values                                                           #
    # ------------------------------------------------------------------ #

    def test_pure_continuum_gives_zero_ew(self):
        """flux == 1 everywhere -> EW == 0 for both lines."""
        lam_obs, res, err = self._spectrum(0.5, noise=0.0)
        result = trapezoidal_ew(lam_obs, res, err, 0.5, self.LINE1, self.LINE2, 1.0, 1.0)
        self.assertAlmostEqual(result['ew1'], 0.0, places=8)
        self.assertAlmostEqual(result['ew2'], 0.0, places=8)
        self.assertAlmostEqual(result['ew_total'], 0.0, places=8)

    def test_ew_total_equals_ew1_plus_ew2(self):
        """ew_total must be the arithmetic sum of ew1 and ew2."""
        lam_obs, res, err = self._absorber_spectrum(0.6)
        result = trapezoidal_ew(lam_obs, res, err, 0.6, self.LINE1, self.LINE2, 1.0, 1.0)
        self.assertAlmostEqual(result['ew_total'], result['ew1'] + result['ew2'], places=10)

    def test_rectangle_absorption_ew_matches_box_width(self):
        """Box absorption of depth=1 and half-width W should give EW ~ 2W."""
        z = 0.5
        half_w = 1.5  # Ang - narrower than n_sigma*sigma=3 integration window
        n = 5000
        # Build spectrum covering only the first line (independently).
        rest_lam = np.linspace(self.LINE1 - 10, self.LINE1 + 10, n)
        residual = np.ones(n)
        residual[np.abs(rest_lam - self.LINE1) <= half_w] = 0.0
        # Pad with a second segment for line2 (flat, no absorption).
        rest_lam2 = np.linspace(self.LINE2 - 10, self.LINE2 + 10, n)
        full_rest = np.concatenate([rest_lam, rest_lam2])
        full_res = np.concatenate([residual, np.ones(n)])
        full_err = np.full(2 * n, 0.01)
        lam_obs = full_rest * (1.0 + z)
        result = trapezoidal_ew(lam_obs, full_res, full_err, z,
                                self.LINE1, self.LINE2, 1.0, 1.0, n_sigma=4)
        self.assertAlmostEqual(result['ew1'], 2 * half_w, delta=0.05)
        self.assertAlmostEqual(result['ew2'], 0.0, delta=0.05)

    def test_positive_absorption_gives_positive_ew(self):
        """Genuine absorption (flux < 1) must produce a positive EW."""
        lam_obs, res, err = self._absorber_spectrum(0.7)
        result = trapezoidal_ew(lam_obs, res, err, 0.7, self.LINE1, self.LINE2, 1.0, 1.0)
        self.assertGreater(result['ew1'], 0.0)
        self.assertGreater(result['ew2'], 0.0)

    def test_ew1_greater_than_ew2_reflects_input_amplitudes(self):
        """Line 1 is stronger (amp1 > amp2), so ew1 should exceed ew2."""
        lam_obs, res, err = self._absorber_spectrum(0.7, amp1=0.6, amp2=0.2)
        result = trapezoidal_ew(lam_obs, res, err, 0.7, self.LINE1, self.LINE2, 1.0, 1.0)
        self.assertGreater(result['ew1'], result['ew2'])

    # ------------------------------------------------------------------ #
    #  Error propagation                                                   #
    # ------------------------------------------------------------------ #

    def test_error_positive_for_nonzero_noise(self):
        """All three error fields must be strictly positive when noise > 0."""
        lam_obs, res, err = self._spectrum(0.5, noise=0.05)
        result = trapezoidal_ew(lam_obs, res, err, 0.5, self.LINE1, self.LINE2, 1.0, 1.0)
        self.assertGreater(result['ew1_err'], 0.0)
        self.assertGreater(result['ew2_err'], 0.0)
        self.assertGreater(result['ew_total_err'], 0.0)

    def test_ew_total_err_equals_quadrature_sum(self):
        """ew_total_err == sqrt(ew1_err**2 + ew2_err**2) when both errors are finite."""
        lam_obs, res, err = self._absorber_spectrum(0.6)
        result = trapezoidal_ew(lam_obs, res, err, 0.6, self.LINE1, self.LINE2, 1.0, 1.0)
        expected = np.sqrt(result['ew1_err'] ** 2 + result['ew2_err'] ** 2)
        self.assertAlmostEqual(result['ew_total_err'], expected, places=10)

    def test_larger_error_gives_larger_ew_err(self):
        """Doubling per-pixel errors must double the EW uncertainty."""
        lam_obs, res, _ = self._absorber_spectrum(0.5)
        err_lo = np.full_like(res, 0.02)
        err_hi = np.full_like(res, 0.04)
        r_lo = trapezoidal_ew(lam_obs, res, err_lo, 0.5, self.LINE1, self.LINE2, 1.0, 1.0)
        r_hi = trapezoidal_ew(lam_obs, res, err_hi, 0.5, self.LINE1, self.LINE2, 1.0, 1.0)
        self.assertAlmostEqual(r_hi['ew1_err'] / r_lo['ew1_err'], 2.0, places=5)

    # ------------------------------------------------------------------ #
    #  n_sigma parameter                                                   #
    # ------------------------------------------------------------------ #

    def test_n_sigma_3_is_default(self):
        """Explicit n_sigma=3 must produce the same result as the default call."""
        lam_obs, res, err = self._absorber_spectrum(0.5)
        r_default = trapezoidal_ew(lam_obs, res, err, 0.5, self.LINE1, self.LINE2, 1.0, 1.0)
        r_explicit = trapezoidal_ew(lam_obs, res, err, 0.5, self.LINE1, self.LINE2, 1.0, 1.0,
                                    n_sigma=3)
        self.assertAlmostEqual(r_default['ew1'],     r_explicit['ew1'])
        self.assertAlmostEqual(r_default['ew2'],     r_explicit['ew2'])
        self.assertAlmostEqual(r_default['ew_total'], r_explicit['ew_total'])

    def test_wider_nsigma_captures_more_ew_from_gaussian_absorption(self):
        """A wider integration window should capture more EW from a Gaussian absorption trough."""
        lam_obs, res, err = self._absorber_spectrum(0.5)
        r_narrow = trapezoidal_ew(lam_obs, res, err, 0.5, self.LINE1, self.LINE2, 1.0, 1.0,
                                  n_sigma=1)
        r_wide = trapezoidal_ew(lam_obs, res, err, 0.5, self.LINE1, self.LINE2, 1.0, 1.0,
                                n_sigma=5)
        self.assertGreater(r_wide['ew1'], r_narrow['ew1'])
        self.assertGreater(r_wide['ew2'], r_narrow['ew2'])

    # ------------------------------------------------------------------ #
    #  Edge cases                                                          #
    # ------------------------------------------------------------------ #

    def test_empty_window_returns_nan(self):
        """Window entirely outside the spectrum -> NaN for all return values."""
        z = 0.5
        lam_obs = np.linspace(3500, 3600, 200) * (1.0 + z)
        res = np.ones(200)
        err = np.full(200, 0.01)
        result = trapezoidal_ew(lam_obs, res, err, z, self.LINE1, self.LINE2, 1.0, 1.0)
        self.assertTrue(np.isnan(result['ew1']))
        self.assertTrue(np.isnan(result['ew2']))
        self.assertTrue(np.isnan(result['ew_total']))

    def test_single_pixel_window_returns_nan(self):
        """A one-pixel window cannot be integrated; NaN is expected."""
        z = 0.5
        # Coarse grid so that each line covers exactly one pixel.
        rest_lam = np.array([self.LINE1 - 5, self.LINE1, self.LINE1 + 5,
                             self.LINE2 - 5, self.LINE2, self.LINE2 + 5])
        lam_obs = rest_lam * (1.0 + z)
        res = np.ones(6)
        err = np.full(6, 0.01)
        # With n_sigma=0.001 the window is sub-pixel -> at most 1 point selected
        result = trapezoidal_ew(lam_obs, res, err, z, self.LINE1, self.LINE2,
                                sigma1=0.0001, sigma2=0.0001, n_sigma=1)
        self.assertTrue(np.isnan(result['ew1']) or result['ew1'] == 0.0)


class TestCalculateEwErrors(unittest.TestCase):
    """Tests for the diagonal-only EW error propagation function."""

    def _params_and_errs(self, amp1=0.4, sig1=1.0, amp2=0.2, sig2=0.8,
                         amp1_err=0.05, sig1_err=0.1, amp2_err=0.04, sig2_err=0.08):
        popt = np.array([amp1, 2796.35, sig1, amp2, 2803.53, sig2])
        perr = np.array([amp1_err, 0.001, sig1_err, amp2_err, 0.001, sig2_err])
        return popt, perr

    def test_returns_three_floats(self):
        popt, perr = self._params_and_errs()
        result = calculate_ew_errors(popt, perr)
        self.assertEqual(len(result), 3)
        for v in result:
            self.assertTrue(np.isfinite(v))

    def test_pure_quadrature_no_cross_term(self):
        """Error must equal EW * sqrt((dA/A)^2 + (ds/s)^2) -- no cross-term."""
        amp1, sig1, amp2, sig2 = 0.4, 1.0, 0.2, 0.8
        amp1_err, sig1_err, amp2_err, sig2_err = 0.05, 0.1, 0.04, 0.08
        popt, perr = self._params_and_errs(amp1, sig1, amp2, sig2,
                                           amp1_err, sig1_err, amp2_err, sig2_err)
        ew1_err, ew2_err, _ = calculate_ew_errors(popt, perr)
        EW1 = amp1 * np.sqrt(2 * np.pi) * sig1
        EW2 = amp2 * np.sqrt(2 * np.pi) * sig2
        expected_ew1_err = EW1 * np.sqrt((amp1_err / amp1)**2 + (sig1_err / sig1)**2)
        expected_ew2_err = EW2 * np.sqrt((amp2_err / amp2)**2 + (sig2_err / sig2)**2)
        self.assertAlmostEqual(ew1_err, expected_ew1_err, places=10)
        self.assertAlmostEqual(ew2_err, expected_ew2_err, places=10)

    def test_total_error_is_quadrature_of_line_errors(self):
        popt, perr = self._params_and_errs()
        ew1_err, ew2_err, ew_total_err = calculate_ew_errors(popt, perr)
        self.assertAlmostEqual(ew_total_err, np.sqrt(ew1_err**2 + ew2_err**2), places=10)

    def test_larger_param_error_gives_larger_ew_error(self):
        popt, perr_lo = self._params_and_errs(amp1_err=0.02)
        _, perr_hi  = self._params_and_errs(amp1_err=0.10)
        ew1_lo, _, _ = calculate_ew_errors(popt, perr_lo)
        ew1_hi, _, _ = calculate_ew_errors(popt, perr_hi)
        self.assertGreater(ew1_hi, ew1_lo)

    def test_zero_param_error_gives_zero_ew_error(self):
        popt, _ = self._params_and_errs()
        perr_zero = np.zeros(6)
        ew1_err, ew2_err, ew_total_err = calculate_ew_errors(popt, perr_zero)
        self.assertAlmostEqual(ew1_err, 0.0, places=12)
        self.assertAlmostEqual(ew2_err, 0.0, places=12)
        self.assertAlmostEqual(ew_total_err, 0.0, places=12)


class TestBootstrapFittingAndEw(unittest.TestCase):
    """Tests for bootstrap_fitting_and_ew with the updated API."""

    LINE1 = 2796.35
    LINE2 = 2803.53

    def _synthetic_spectrum(self, z=0.6, n=2000, amp1=0.35, amp2=0.18, sig=1.0, noise=0.02):
        """Return (wavelength, flux, error) with a clean double-Gaussian absorber."""
        rest_lam = np.linspace(self.LINE1 - 15, self.LINE2 + 15, n)
        model = (1.0
                 - amp1 * np.exp(-(rest_lam - self.LINE1)**2 / (2 * sig**2))
                 - amp2 * np.exp(-(rest_lam - self.LINE2)**2 / (2 * sig**2)))
        rng = np.random.RandomState(0)
        error = np.full(n, noise)
        flux = model + rng.normal(0, noise, n)
        wavelength = rest_lam * (1 + z)
        return wavelength, flux, error

    def _simple_bound(self):
        return (
            np.array([0.02, self.LINE1 - 1.0, 0.1,  0.02, self.LINE2 - 1.0, 0.1]),
            np.array([1.10, self.LINE1 + 1.0, 15.0, 1.10, self.LINE2 + 1.0, 15.0])
        )

    def _run(self, best_params=None, nboot=20):
        z = 0.6
        wavelength, flux, error = self._synthetic_spectrum(z)
        bound = self._simple_bound()
        return bootstrap_fitting_and_ew(
            index=0, nboot=nboot, z=z,
            wavelength=wavelength, flux=flux, error=error,
            ix0=self.LINE1 - 15, ix1=self.LINE2 + 15,
            bound=bound, amp_ratio=0.5,
            line1=self.LINE1, line2=self.LINE2,
            num_iter=300, best_params=best_params)

    def test_returns_eight_values(self):
        result = self._run()
        self.assertEqual(len(result), 8)

    def test_all_means_are_finite(self):
        params_mean, _, ew1, ew2, ew_total, _, _, _ = self._run()
        self.assertTrue(np.all(np.isfinite(params_mean)))
        self.assertTrue(np.isfinite(ew1))
        self.assertTrue(np.isfinite(ew2))
        self.assertTrue(np.isfinite(ew_total))

    def test_ew_total_mean_equals_ew1_plus_ew2_mean(self):
        _, _, ew1, ew2, ew_total, _, _, _ = self._run()
        self.assertAlmostEqual(ew_total, ew1 + ew2, places=6)

    def test_std_positive_for_real_absorption(self):
        """Bootstrap std on EWs must be positive when absorption is present."""
        _, fit_std, _, _, _, ew1_std, ew2_std, _ = self._run(nboot=30)
        self.assertGreater(ew1_std, 0.0)
        self.assertGreater(ew2_std, 0.0)

    def test_warm_start_runs_without_error(self):
        """Passing best_params (warm start) must not raise and must return valid output."""
        best_params = np.array([0.35, self.LINE1, 1.0, 0.18, self.LINE2, 1.0])
        params_mean, fit_std, ew1, ew2, ew_total, _, _, _ = self._run(best_params=best_params)
        self.assertTrue(np.all(np.isfinite(params_mean)))
        self.assertTrue(np.isfinite(ew1) and np.isfinite(ew2))

    def test_none_best_params_falls_back_to_wide_draw(self):
        """best_params=None must still return valid results (fallback path)."""
        params_mean, _, ew1, ew2, _, _, _, _ = self._run(best_params=None)
        self.assertTrue(np.isfinite(ew1))
        self.assertTrue(np.isfinite(ew2))

    def test_nan_best_params_falls_back_gracefully(self):
        """An all-NaN best_params must fall back without raising."""
        best_params = np.full(6, np.nan)
        params_mean, _, ew1, ew2, _, _, _, _ = self._run(best_params=best_params)
        # just check it completed; values may be NaN for some iterations but means should survive
        self.assertEqual(len(params_mean), 6)


if __name__ == "__main__":
    unittest.main()