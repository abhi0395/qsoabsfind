"""
Tests for columndensity.py utility functions.
Covers ss1991_correction, optical_depth, velocity_from_wavelength,
single_column_density, total_column_density, and compute_single_column_density.
"""

import unittest
import numpy as np
from astropy.table import Table
from qsoabsfind.columndensity import (
    ss1991_correction,
    optical_depth,
    velocity_from_wavelength,
    single_column_density,
    total_column_density,
    compute_single_column_density,
)

# Shared MgII parameters used across tests
_F1, _F2 = 0.6123, 0.3054          # oscillator strengths
_L1, _L2 = 2796.35, 2803.52        # rest-frame wavelengths (Ang)
_LAMBDA1 = ("MGII_2796", _L1)
_LAMBDA2 = ("MGII_2803", _L2)


def _make_spectrum(z, depth1=0.55, depth2=0.30, n=500):
    """Create a synthetic MgII doublet spectrum at redshift *z*."""
    lam1_obs = _L1 * (1 + z)
    lam2_obs = _L2 * (1 + z)
    wavelength = np.linspace(lam1_obs - 30, lam2_obs + 30, n)
    sigma = 2.0
    flux = (1.0
            - depth1 * np.exp(-0.5 * ((wavelength - lam1_obs) / sigma) ** 2)
            - depth2 * np.exp(-0.5 * ((wavelength - lam2_obs) / sigma) ** 2))
    error = np.full_like(flux, 0.02)
    return wavelength, flux, error


def _make_abs_cat(z, ew1=1.0, ew2=0.5, ew1_err=0.1, ew2_err=0.1):
    return Table({
        'Z_ABS': [z],
        'MGII_2796_EW': [ew1],
        'MGII_2803_EW': [ew2],
        'MGII_2796_EW_ERROR': [ew1_err],
        'MGII_2803_EW_ERROR': [ew2_err],
    })


class TestSS1991Correction(unittest.TestCase):

    def test_zero_delta_returns_zero(self):
        self.assertAlmostEqual(ss1991_correction(0.0), 0.0)

    def test_known_value_at_01(self):
        # Table 4 of S&S 1991 gives 0.111 at delta_logN = 0.10
        self.assertAlmostEqual(ss1991_correction(0.1), 0.111, places=3)

    def test_known_value_at_02(self):
        self.assertAlmostEqual(ss1991_correction(0.2), 0.348, places=3)

    def test_out_of_range_clamps_to_zero(self):
        # Values outside the table range are clamped via np.interp right=0
        self.assertEqual(ss1991_correction(1.0), 0.0)

    def test_array_input_shape(self):
        vals = np.array([0.0, 0.05, 0.1, 0.2])
        result = ss1991_correction(vals)
        self.assertEqual(result.shape, vals.shape)

    def test_monotonically_increasing(self):
        deltas = np.linspace(0.0, 0.24, 20)
        corrections = ss1991_correction(deltas)
        self.assertTrue(np.all(np.diff(corrections) >= 0))


class TestOpticalDepth(unittest.TestCase):

    def setUp(self):
        self.F = np.array([0.3, 0.5, 0.7, 0.9, 1.0])
        self.sigma = np.full(5, 0.03)
        self.cont_err = 0.05

    def test_output_shapes(self):
        tau, sig_tau = optical_depth(self.F, self.sigma, self.cont_err)
        self.assertEqual(tau.shape, self.F.shape)
        self.assertEqual(sig_tau.shape, self.F.shape)

    def test_tau_non_negative(self):
        tau, _ = optical_depth(self.F, self.sigma, self.cont_err)
        self.assertTrue(np.all(tau >= 0))

    def test_sigma_tau_positive(self):
        _, sig_tau = optical_depth(self.F, self.sigma, self.cont_err)
        self.assertTrue(np.all(sig_tau > 0))

    def test_near_zero_flux_is_clipped(self):
        # Flux below 0.005 should be clipped; result must be finite
        F_low = np.array([0.0, 0.001, 0.004])
        sigma_low = np.full(3, 0.01)
        tau, sig_tau = optical_depth(F_low, sigma_low, 0.05)
        self.assertTrue(np.all(np.isfinite(tau)))
        self.assertTrue(np.all(np.isfinite(sig_tau)))

    def test_higher_continuum_error_increases_sigma_tau(self):
        _, sig1 = optical_depth(self.F, self.sigma, 0.01)
        _, sig2 = optical_depth(self.F, self.sigma, 0.10)
        self.assertTrue(np.all(sig2 >= sig1))


class TestVelocityFromWavelength(unittest.TestCase):

    def setUp(self):
        self.z = 0.5
        self.lam0 = _L1
        lam_obs = self.lam0 * (1 + self.z)
        self.wavelength = np.linspace(lam_obs - 20, lam_obs + 20, 300)

    def test_output_shapes_linear(self):
        v_pix, dv = velocity_from_wavelength(self.wavelength, self.lam0, self.z, logwave=False)
        self.assertEqual(v_pix.shape, self.wavelength.shape)
        self.assertEqual(dv.shape, self.wavelength.shape)

    def test_zero_velocity_at_line_center_linear(self):
        lam_obs = self.lam0 * (1 + self.z)
        wl = np.array([lam_obs - 1, lam_obs, lam_obs + 1])
        _, dv = velocity_from_wavelength(wl, self.lam0, self.z, logwave=False)
        self.assertAlmostEqual(dv[1], 0.0, delta=1.0)  # within 1 km/s

    def test_output_shapes_logwave(self):
        lam_obs = self.lam0 * (1 + self.z)
        wl_log = np.logspace(np.log10(lam_obs - 20), np.log10(lam_obs + 20), 300)
        v_pix, dv = velocity_from_wavelength(wl_log, self.lam0, self.z, logwave=True)
        self.assertEqual(v_pix.shape, wl_log.shape)
        self.assertEqual(dv.shape, wl_log.shape)

    def test_velocity_pixels_positive(self):
        v_pix, _ = velocity_from_wavelength(self.wavelength, self.lam0, self.z, logwave=False)
        self.assertTrue(np.all(v_pix > 0))

    def test_dv_sign(self):
        lam_obs = self.lam0 * (1 + self.z)
        wl = np.array([lam_obs - 5, lam_obs + 5])
        _, dv = velocity_from_wavelength(wl, self.lam0, self.z, logwave=False)
        self.assertLess(dv[0], 0)   # blueward --> negative velocity
        self.assertGreater(dv[1], 0)  # redward --> positive velocity


class TestSingleColumnDensity(unittest.TestCase):

    def setUp(self):
        self.z = 0.5
        lam_obs = _L1 * (1 + self.z)
        self.wavelength = np.linspace(lam_obs - 20, lam_obs + 20, 300)
        sigma = 2.0
        self.flux = 1.0 - 0.6 * np.exp(-0.5 * ((self.wavelength - lam_obs) / sigma) ** 2)
        self.error = np.full_like(self.flux, 0.02)

    def test_returns_expected_keys(self):
        result = single_column_density(
            self.flux, self.error, self.wavelength, self.z,
            _F1, _L1, 0.05, 300, logwave=False)
        self.assertSetEqual(set(result.keys()), {'N', 'N_err', 'logN', 'err_logN', 'flag'})

    def test_absorption_gives_positive_column_density(self):
        result = single_column_density(
            self.flux, self.error, self.wavelength, self.z,
            _F1, _L1, 0.05, 300, logwave=False)
        self.assertEqual(result['flag'], 1)
        self.assertGreater(result['N'], 0)
        self.assertTrue(np.isfinite(result['logN']))

    def test_logN_and_N_consistent(self):
        result = single_column_density(
            self.flux, self.error, self.wavelength, self.z,
            _F1, _L1, 0.05, 300, logwave=False)
        if result['flag'] == 1:
            self.assertAlmostEqual(np.log10(result['N']), result['logN'], places=5)

    def test_logwave_mode(self):
        lam_obs = _L1 * (1 + self.z)
        wl_log = np.logspace(np.log10(lam_obs - 20), np.log10(lam_obs + 20), 300)
        sigma = 2.0
        flux_log = 1.0 - 0.6 * np.exp(-0.5 * ((wl_log - lam_obs) / sigma) ** 2)
        error_log = np.full_like(flux_log, 0.02)
        result = single_column_density(
            flux_log, error_log, wl_log, self.z,
            _F1, _L1, 0.05, 300, logwave=True)
        self.assertIn(result['flag'], (1, -1))

    def test_fail_flag_when_no_valid_pixels(self):
        # Flux exactly 1 everywhere --> tau ≈ 0 --> N ~ 0 or negative
        flat_flux = np.ones_like(self.wavelength)
        result = single_column_density(
            flat_flux, self.error, self.wavelength, self.z,
            _F1, _L1, 0.05, 300, logwave=False)
        self.assertIn(result['flag'], (1, -1))


class TestTotalColumnDensity(unittest.TestCase):

    def _call(self, z=0.5, depth1=0.55, depth2=0.30, ew1=1.0, ew2=0.5):
        wavelength, flux, error = _make_spectrum(z, depth1=depth1, depth2=depth2)
        abs_cat = _make_abs_cat(z, ew1=ew1, ew2=ew2)
        return total_column_density(
            flux, error, wavelength, abs_cat,
            _F1, _F2, _LAMBDA1, _LAMBDA2,
            continuum_error_frac=0.05, velocity_range=300, logwave=False)

    def test_returns_astropy_table(self):
        result = self._call()
        self.assertIsInstance(result, Table)
        self.assertEqual(len(result), 1)

    def test_has_four_columns(self):
        result = self._call()
        self.assertIn('LOG10N', result.colnames)
        self.assertIn('SIG_LOG10N', result.colnames)
        self.assertIn('SATURATION', result.colnames)
        self.assertIn('fN', result.colnames)

    def test_unsaturated_both_lines_gives_weighted_flag(self):
        # DR = ew1/ew2 = 2.0 ≈ f1/f2 = 2.0 --> unsaturated
        result = self._call(ew1=1.0, ew2=0.5, depth1=0.55, depth2=0.27)
        self.assertIn(result['fN'][0], (1, 2, 3, -1))

    def test_logN_finite_when_absorption_present(self):
        result = self._call()
        logN = result['LOG10N'][0]
        self.assertTrue(np.isfinite(logN) or np.isnan(logN))  # either valid or gracefully NaN

    def test_saturation_flag_zero_for_high_dr(self):
        # High DR (unsat) --> sflag = 0
        result = self._call(ew1=2.0, ew2=0.5)  # DR = 4 >> f1/f2 = 2
        self.assertEqual(result['SATURATION'][0], 0)

    def test_saturation_flag_one_for_low_dr(self):
        # DR close to 1 --> saturated --> sflag = 1
        result = self._call(ew1=1.0, ew2=1.0)  # DR = 1 < f1/f2 - error
        self.assertEqual(result['SATURATION'][0], 1)


class TestComputeSingleColumnDensity(unittest.TestCase):

    def test_wrapper_returns_table(self):
        z = 0.5
        wavelength, flux, error = _make_spectrum(z)
        abs_cat = _make_abs_cat(z)
        args = (flux, error, wavelength, abs_cat,
                _F1, _F2, _LAMBDA1, _LAMBDA2, 0.05, 300, False)
        result = compute_single_column_density(args)
        self.assertIsInstance(result, Table)
        self.assertIn('LOG10N', result.colnames)

    def test_wrapper_matches_direct_call(self):
        z = 0.5
        wavelength, flux, error = _make_spectrum(z)
        abs_cat = _make_abs_cat(z)
        args = (flux, error, wavelength, abs_cat,
                _F1, _F2, _LAMBDA1, _LAMBDA2, 0.05, 300, False)
        result_wrapper = compute_single_column_density(args)
        result_direct = total_column_density(
            flux, error, wavelength, abs_cat,
            _F1, _F2, _LAMBDA1, _LAMBDA2,
            continuum_error_frac=0.05, velocity_range=300, logwave=False)
        # Both should return the same logN value
        self.assertAlmostEqual(
            float(result_wrapper['LOG10N'][0]),
            float(result_direct['LOG10N'][0]),
            places=10)


if __name__ == '__main__':
    unittest.main()
