"""
Tests for columndensity.py.

Updated for the revised AODM implementation:
- continuum-placement uncertainty is propagated in single_column_density(),
  not inside optical_depth();
- saturated pixels are retained at the flux floor and flagged;
- flux above unity is not clipped;
- S&S correction is valid only for 0 <= delta_logN <= 0.24;
- saturation is diagnosed from AOD column-density disagreement, not EW ratio;
- total_column_density returns additional diagnostic columns.
"""

import unittest
from unittest.mock import patch

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


# ----------------------------------------------------------------------
# Shared MgII parameters
# ----------------------------------------------------------------------

_F1, _F2 = 0.6123, 0.3054
_L1, _L2 = 2796.35, 2803.52
_LAMBDA1 = ("MGII_2796", _L1)
_LAMBDA2 = ("MGII_2803", _L2)


def _make_spectrum(z, depth1=0.55, depth2=0.30, n=800):
    """Create a synthetic MgII doublet spectrum."""
    lam1_obs = _L1 * (1 + z)
    lam2_obs = _L2 * (1 + z)

    wavelength = np.linspace(
        lam1_obs - 30.0,
        lam2_obs + 30.0,
        n,
    )

    sigma = 2.0

    flux = (
        1.0
        - depth1
        * np.exp(
            -0.5
            * ((wavelength - lam1_obs) / sigma) ** 2
        )
        - depth2
        * np.exp(
            -0.5
            * ((wavelength - lam2_obs) / sigma) ** 2
        )
    )

    error = np.full_like(flux, 0.02)

    return wavelength, flux, error


def _make_abs_cat(z):
    """
    total_column_density() now uses only Z_ABS from the absorber row.
    EW columns are no longer required for saturation classification.
    """
    return Table({"Z_ABS": [z]})[0]


def _fake_single_result(
    N,
    N_err_stat,
    *,
    lower=False,
    n_saturated=0,
    cont_frac=0.02,
):
    """Construct a deterministic single-line result for total-column tests."""
    N = float(N)
    N_err_stat = float(N_err_stat)

    N_cont_plus = N * (1.0 + cont_frac)
    N_cont_minus = N * (1.0 - cont_frac)
    N_err_cont = cont_frac * N
    N_err = np.sqrt(N_err_stat**2 + N_err_cont**2)

    return {
        "N": N,
        "N_err": N_err,
        "N_err_stat": N_err_stat,
        "N_err_cont": N_err_cont,
        "N_cont_plus": N_cont_plus,
        "N_cont_minus": N_cont_minus,
        "logN": np.log10(N),
        "err_logN": N_err / (N * np.log(10.0)),
        "flag": 1,
        "is_lower_limit": bool(lower),
        "n_saturated": int(n_saturated),
        "n_pixels": 20,
    }


# ======================================================================
# Savage & Sembach correction
# ======================================================================

class TestSS1991Correction(unittest.TestCase):

    def test_zero_delta_returns_zero(self):
        self.assertAlmostEqual(
            ss1991_correction(0.0),
            0.0,
        )

    def test_known_value_at_01(self):
        self.assertAlmostEqual(
            ss1991_correction(0.10),
            0.111,
            places=3,
        )

    def test_known_value_at_02(self):
        self.assertAlmostEqual(
            ss1991_correction(0.20),
            0.348,
            places=3,
        )

    def test_upper_table_value(self):
        self.assertAlmostEqual(
            ss1991_correction(0.24),
            0.600,
            places=3,
        )

    def test_out_of_range_returns_nan(self):
        self.assertTrue(
            np.isnan(
                ss1991_correction(0.25)
            )
        )
        self.assertTrue(
            np.isnan(
                ss1991_correction(-0.01)
            )
        )

    def test_return_slope(self):
        corr, slope = ss1991_correction(
            0.10,
            return_slope=True,
        )

        self.assertAlmostEqual(
            corr,
            0.111,
            places=3,
        )

        self.assertTrue(
            np.isfinite(slope)
        )

    def test_monotonically_increasing(self):
        deltas = np.linspace(
            0.0,
            0.24,
            25,
        )

        corrections = np.array([
            ss1991_correction(x)
            for x in deltas
        ])

        self.assertTrue(
            np.all(
                np.diff(corrections) >= 0
            )
        )


# ======================================================================
# Optical depth
# ======================================================================

class TestOpticalDepth(unittest.TestCase):

    def setUp(self):
        self.F = np.array([
            0.3,
            0.5,
            0.7,
            0.9,
            1.0,
        ])

        self.sigma = np.full(
            5,
            0.03,
        )

    def test_output_shapes(self):
        tau, sig_tau = optical_depth(
            self.F,
            self.sigma,
            0.05,
        )

        self.assertEqual(
            tau.shape,
            self.F.shape,
        )

        self.assertEqual(
            sig_tau.shape,
            self.F.shape,
        )

    def test_tau_matches_minus_log_flux(self):
        tau, _ = optical_depth(
            self.F,
            self.sigma,
            0.05,
        )

        np.testing.assert_allclose(
            tau,
            -np.log(self.F),
        )

    def test_sigma_tau_matches_error_over_flux(self):
        _, sig_tau = optical_depth(
            self.F,
            self.sigma,
            0.05,
        )

        np.testing.assert_allclose(
            sig_tau,
            self.sigma / self.F,
        )

    def test_near_zero_flux_is_clipped(self):
        F_low = np.array([
            0.0,
            0.001,
            0.004,
        ])

        sigma_low = np.full(
            3,
            0.01,
        )

        tau, sig_tau = optical_depth(
            F_low,
            sigma_low,
            0.05,
        )

        self.assertTrue(
            np.all(
                np.isfinite(tau)
            )
        )

        self.assertTrue(
            np.all(
                np.isfinite(sig_tau)
            )
        )

    def test_flux_above_unity_gives_negative_tau(self):
        F = np.array([
            0.9,
            1.0,
            1.1,
        ])

        sigma = np.full(
            3,
            0.02,
        )

        tau, _ = optical_depth(
            F,
            sigma,
            0.05,
        )

        self.assertGreater(
            tau[0],
            0.0,
        )

        self.assertAlmostEqual(
            tau[1],
            0.0,
        )

        self.assertLess(
            tau[2],
            0.0,
        )

    def test_continuum_error_not_added_pixelwise(self):
        _, sig1 = optical_depth(
            self.F,
            self.sigma,
            0.01,
        )

        _, sig2 = optical_depth(
            self.F,
            self.sigma,
            0.10,
        )

        np.testing.assert_allclose(
            sig1,
            sig2,
        )


# ======================================================================
# Velocity conversion
# ======================================================================

class TestVelocityFromWavelength(unittest.TestCase):

    def setUp(self):
        self.z = 0.5
        self.lam0 = _L1

        lam_obs = (
            self.lam0
            * (1 + self.z)
        )

        self.wavelength = np.linspace(
            lam_obs - 20,
            lam_obs + 20,
            300,
        )

    def test_output_shapes_linear(self):
        dv_pixel, velocity = velocity_from_wavelength(
            self.wavelength,
            self.lam0,
            self.z,
            logwave=False,
        )

        self.assertEqual(
            dv_pixel.shape,
            self.wavelength.shape,
        )

        self.assertEqual(
            velocity.shape,
            self.wavelength.shape,
        )

    def test_zero_velocity_at_line_center(self):
        lam_obs = (
            self.lam0
            * (1 + self.z)
        )

        wl = np.array([
            lam_obs - 1,
            lam_obs,
            lam_obs + 1,
        ])

        _, velocity = velocity_from_wavelength(
            wl,
            self.lam0,
            self.z,
            logwave=False,
        )

        self.assertAlmostEqual(
            velocity[1],
            0.0,
            places=10,
        )

    def test_output_shapes_logwave(self):
        lam_obs = (
            self.lam0
            * (1 + self.z)
        )

        wl_log = np.logspace(
            np.log10(lam_obs - 20),
            np.log10(lam_obs + 20),
            300,
        )

        dv_pixel, velocity = velocity_from_wavelength(
            wl_log,
            self.lam0,
            self.z,
            logwave=True,
        )

        self.assertEqual(
            dv_pixel.shape,
            wl_log.shape,
        )

        self.assertEqual(
            velocity.shape,
            wl_log.shape,
        )

    def test_velocity_pixels_positive(self):
        dv_pixel, _ = velocity_from_wavelength(
            self.wavelength,
            self.lam0,
            self.z,
            logwave=False,
        )

        self.assertTrue(
            np.all(
                dv_pixel > 0
            )
        )

    def test_velocity_sign(self):
        lam_obs = (
            self.lam0
            * (1 + self.z)
        )

        wl = np.array([
            lam_obs - 5,
            lam_obs + 5,
        ])

        _, velocity = velocity_from_wavelength(
            wl,
            self.lam0,
            self.z,
            logwave=False,
        )

        self.assertLess(
            velocity[0],
            0,
        )

        self.assertGreater(
            velocity[1],
            0,
        )


# ======================================================================
# Single-line AODM
# ======================================================================

class TestSingleColumnDensity(unittest.TestCase):

    def setUp(self):
        self.z = 0.5

        lam_obs = (
            _L1
            * (1 + self.z)
        )

        self.wavelength = np.linspace(
            lam_obs - 20,
            lam_obs + 20,
            500,
        )

        sigma = 2.0

        self.flux = (
            1.0
            - 0.6
            * np.exp(
                -0.5
                * (
                    (
                        self.wavelength
                        - lam_obs
                    )
                    / sigma
                ) ** 2
            )
        )

        self.error = np.full_like(
            self.flux,
            0.02,
        )

    def _call(self, flux=None, cont_err=0.05):
        if flux is None:
            flux = self.flux

        return single_column_density(
            flux,
            self.error,
            self.wavelength,
            self.z,
            _F1,
            _L1,
            cont_err,
            150,
            logwave=False,
        )

    def test_returns_expected_keys(self):
        result = self._call()

        expected = {
            "N",
            "N_err",
            "N_err_stat",
            "N_err_cont",
            "N_cont_plus",
            "N_cont_minus",
            "logN",
            "err_logN",
            "flag",
            "is_lower_limit",
            "n_saturated",
            "n_pixels",
        }

        self.assertSetEqual(
            set(result.keys()),
            expected,
        )

    def test_absorption_gives_positive_column_density(self):
        result = self._call()

        self.assertEqual(
            result["flag"],
            1,
        )

        self.assertGreater(
            result["N"],
            0,
        )

        self.assertTrue(
            np.isfinite(
                result["logN"]
            )
        )

    def test_logN_and_N_consistent(self):
        result = self._call()

        self.assertAlmostEqual(
            np.log10(
                result["N"]
            ),
            result["logN"],
            places=10,
        )

    def test_total_error_contains_statistical_error(self):
        result = self._call()

        self.assertGreaterEqual(
            result["N_err"],
            result["N_err_stat"],
        )

    def test_continuum_error_increases_total_error(self):
        result0 = self._call(
            cont_err=0.0
        )

        result5 = self._call(
            cont_err=0.05
        )

        self.assertGreater(
            result5["N_err"],
            result0["N_err"],
        )

        self.assertGreater(
            result5["N_err_cont"],
            0.0,
        )

    def test_floor_pixel_is_retained_and_flagged(self):
        flux = self.flux.copy()

        center = np.argmin(
            np.abs(
                self.wavelength
                - _L1
                * (1 + self.z)
            )
        )

        flux[center] = 0.0

        result = self._call(
            flux=flux
        )

        self.assertEqual(
            result["flag"],
            1,
        )

        self.assertTrue(
            result["is_lower_limit"]
        )

        self.assertGreaterEqual(
            result["n_saturated"],
            1,
        )

    def test_flat_continuum_fails_column_measurement(self):
        flat_flux = np.ones_like(
            self.wavelength
        )

        result = self._call(
            flux=flat_flux
        )

        self.assertEqual(
            result["flag"],
            -1,
        )

        self.assertTrue(
            np.isnan(
                result["N"]
            )
        )

    def test_logwave_mode(self):
        lam_obs = (
            _L1
            * (1 + self.z)
        )

        wl_log = np.logspace(
            np.log10(lam_obs - 20),
            np.log10(lam_obs + 20),
            500,
        )

        sigma = 2.0

        flux_log = (
            1.0
            - 0.6
            * np.exp(
                -0.5
                * (
                    (
                        wl_log
                        - lam_obs
                    )
                    / sigma
                ) ** 2
            )
        )

        error_log = np.full_like(
            flux_log,
            0.02,
        )

        result = single_column_density(
            flux_log,
            error_log,
            wl_log,
            self.z,
            _F1,
            _L1,
            0.05,
            150,
            logwave=True,
        )

        self.assertEqual(
            result["flag"],
            1,
        )


# ======================================================================
# Doublet combination logic
# ======================================================================

class TestTotalColumnDensity(unittest.TestCase):

    def setUp(self):
        self.z = 0.5
        self.wavelength, self.flux, self.error = _make_spectrum(
            self.z
        )
        self.abs_cat = _make_abs_cat(
            self.z
        )

    def _call_with_single_results(self, strong_result, weak_result):
        """
        MgII 2796 is stronger than 2803, so the first mocked return value
        corresponds to the strong line and the second to the weak line.
        """
        with patch(
            "qsoabsfind.columndensity.single_column_density",
            side_effect=[
                strong_result,
                weak_result,
            ],
        ):
            return total_column_density(
                self.flux,
                self.error,
                self.wavelength,
                self.abs_cat,
                _F1,
                _F2,
                _LAMBDA1,
                _LAMBDA2,
                continuum_error_frac=0.05,
                velocity_range=150,
                logwave=False,
            )

    def test_returns_astropy_table(self):
        result = self._call_with_single_results(
            _fake_single_result(1.0e13, 2.0e11),
            _fake_single_result(1.01e13, 2.0e11),
        )

        self.assertIsInstance(
            result,
            Table,
        )

        self.assertEqual(
            len(result),
            1,
        )

    def test_has_expected_columns(self):
        result = self._call_with_single_results(
            _fake_single_result(1.0e13, 2.0e11),
            _fake_single_result(1.01e13, 2.0e11),
        )

        expected = {
            "LOG10N",
            "SIG_LOG10N",
            "SATURATION",
            "fN",
            "LOWER_LIMIT",
            "DELTA_LOGN",
            "SIG_DELTA_LOGN",
            "NPIX_SAT_STRONG",
            "NPIX_SAT_WEAK",
        }

        self.assertSetEqual(
            set(result.colnames),
            expected,
        )

    def test_consistent_doublet_is_weighted(self):
        strong = _fake_single_result(
            1.00e13,
            2.0e11,
        )

        weak = _fake_single_result(
            1.01e13,
            2.0e11,
        )

        result = self._call_with_single_results(
            strong,
            weak,
        )

        self.assertEqual(
            result["SATURATION"][0],
            0,
        )

        self.assertEqual(
            result["fN"][0],
            1,
        )

        self.assertEqual(
            result["LOWER_LIMIT"][0],
            0,
        )

    def test_significant_positive_delta_applies_ss_correction(self):
        strong = _fake_single_result(
            1.00e13,
            5.0e10,
        )

        # delta_logN = 0.10 dex
        weak = _fake_single_result(
            10.0**13.10,
            5.0e10,
        )

        result = self._call_with_single_results(
            strong,
            weak,
        )

        self.assertEqual(
            result["SATURATION"][0],
            1,
        )

        self.assertEqual(
            result["fN"][0],
            4,
        )

        expected_logN = (
            13.10
            + ss1991_correction(0.10)
        )

        self.assertAlmostEqual(
            result["LOG10N"][0],
            expected_logN,
            places=6,
        )

    def test_delta_above_ss_range_is_lower_limit(self):
        strong = _fake_single_result(
            1.00e13,
            5.0e10,
        )

        weak = _fake_single_result(
            10.0**13.30,
            5.0e10,
        )

        result = self._call_with_single_results(
            strong,
            weak,
        )

        self.assertEqual(
            result["SATURATION"][0],
            2,
        )

        self.assertEqual(
            result["fN"][0],
            5,
        )

        self.assertEqual(
            result["LOWER_LIMIT"][0],
            1,
        )

        self.assertAlmostEqual(
            result["LOG10N"][0],
            np.log10(
                weak["N"]
            ),
            places=10,
        )

    def test_floor_saturation_returns_weak_lower_limit(self):
        strong = _fake_single_result(
            1.0e13,
            2.0e11,
            lower=True,
            n_saturated=2,
        )

        weak = _fake_single_result(
            1.2e13,
            2.0e11,
        )

        result = self._call_with_single_results(
            strong,
            weak,
        )

        self.assertEqual(
            result["SATURATION"][0],
            2,
        )

        self.assertEqual(
            result["fN"][0],
            5,
        )

        self.assertEqual(
            result["LOWER_LIMIT"][0],
            1,
        )

    def test_significantly_negative_delta_is_inconsistent(self):
        strong = _fake_single_result(
            1.20e13,
            5.0e10,
        )

        weak = _fake_single_result(
            1.00e13,
            5.0e10,
        )

        result = self._call_with_single_results(
            strong,
            weak,
        )

        self.assertEqual(
            result["SATURATION"][0],
            -2,
        )

        self.assertEqual(
            result["fN"][0],
            7,
        )

        self.assertTrue(
            np.isnan(
                result["LOG10N"][0]
            )
        )

    def test_saved_delta_logN_matches_inputs(self):
        strong = _fake_single_result(
            1.0e13,
            1.0e11,
        )

        weak = _fake_single_result(
            10.0**13.08,
            1.0e11,
        )

        result = self._call_with_single_results(
            strong,
            weak,
        )

        self.assertAlmostEqual(
            result["DELTA_LOGN"][0],
            0.08,
            places=10,
        )


# ======================================================================
# Multiprocessing wrapper
# ======================================================================

class TestComputeSingleColumnDensity(unittest.TestCase):

    def test_wrapper_returns_table(self):
        z = 0.5
        wavelength, flux, error = _make_spectrum(
            z
        )
        abs_cat = _make_abs_cat(
            z
        )

        args = (
            flux,
            error,
            wavelength,
            abs_cat,
            _F1,
            _F2,
            _LAMBDA1,
            _LAMBDA2,
            0.05,
            150,
            False,
        )

        result = compute_single_column_density(
            args
        )

        self.assertIsInstance(
            result,
            Table,
        )

        self.assertIn(
            "LOG10N",
            result.colnames,
        )

    def test_wrapper_matches_direct_call(self):
        z = 0.5

        wavelength, flux, error = _make_spectrum(
            z
        )

        abs_cat = _make_abs_cat(
            z
        )

        args = (
            flux,
            error,
            wavelength,
            abs_cat,
            _F1,
            _F2,
            _LAMBDA1,
            _LAMBDA2,
            0.05,
            150,
            False,
        )

        result_wrapper = compute_single_column_density(
            args
        )

        result_direct = total_column_density(
            flux,
            error,
            wavelength,
            abs_cat,
            _F1,
            _F2,
            _LAMBDA1,
            _LAMBDA2,
            continuum_error_frac=0.05,
            velocity_range=150,
            logwave=False,
        )

        w = float(
            result_wrapper["LOG10N"][0]
        )
        d = float(
            result_direct["LOG10N"][0]
        )

        if np.isnan(w) and np.isnan(d):
            self.assertTrue(True)
        else:
            self.assertAlmostEqual(
                w,
                d,
                places=10,
            )


if __name__ == "__main__":
    unittest.main()
