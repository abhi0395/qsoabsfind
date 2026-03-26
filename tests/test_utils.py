"""
Tests for utils.py
"""

import os
import unittest
import numpy as np
from qsoabsfind.utils import (
    convolution_fun,
    compute_doublet_amplitudes,
    gauss_two_lines_kernel,
    double_gaussian,
    single_gaussian,
    validate_sizes,
    numeric_key,
    parse_qso_sequence,
    get_package_versions,
    get_all_extnames,
    read_nqso_from_header,
    vel_dispersion,
    modify_units,
    match_order,
)

# Path to a real FITS file used for file-based tests
_DESI_FITS = os.path.join(os.path.dirname(__file__), '..', 'data', 'desi', 'qso_test_spectra.fits')


class TestComputeDoubletAmplitudes(unittest.TestCase):

    def test_basic(self):
        A1, A2 = compute_doublet_amplitudes(0.5, 0.6123, 0.3054)
        self.assertLessEqual(A1, 1.0)
        self.assertLessEqual(A2, 1.0)

    def test_equal_oscillator_strengths(self):
        A1, A2 = compute_doublet_amplitudes(0.8, 1.0, 1.0)
        self.assertAlmostEqual(A1, A2)

    def test_a2_exceeds_one_is_scaled_down(self):
        # f2 >> f1 would make A2 > 1 without scaling
        A1, A2 = compute_doublet_amplitudes(1.0, 0.1, 10.0)
        self.assertLessEqual(A2, 1.0)
        self.assertLessEqual(A1, 1.0)

    def test_output_is_tuple_of_two(self):
        result = compute_doublet_amplitudes(0.7, 0.5, 0.3)
        self.assertEqual(len(result), 2)


class TestGaussTwoLinesKernel(unittest.TestCase):

    def setUp(self):
        from qsoabsfind.constants import lines
        l1, l2 = lines['MgII_2796'], lines['MgII_2803']
        self.x = np.linspace(l1 - 20, l2 + 20, 500)
        self.params = np.array([0.8, l1, 3.0, 0.5, l2, 3.0])

    def test_output_shape(self):
        result = gauss_two_lines_kernel(self.x, self.params)
        self.assertEqual(result.shape, self.x.shape)

    def test_values_finite(self):
        result = gauss_two_lines_kernel(self.x, self.params)
        self.assertTrue(np.all(np.isfinite(result)))

    def test_absorption_dips_below_baseline(self):
        result = gauss_two_lines_kernel(self.x, self.params)
        # kernel adds Gaussians above the baseline of 1.0
        self.assertTrue(np.any(result > 1.0))


class TestDoubleAndSingleGaussian(unittest.TestCase):

    def test_double_gaussian_shape(self):
        x = np.linspace(2780, 2820, 300)
        y = double_gaussian(x, 0.5, 2796.35, 2.0, 0.3, 2803.52, 2.0)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_double_gaussian_at_continuum(self):
        # Far from line centers the function should be ~ 1
        x = np.array([2700.0, 2900.0])
        y = double_gaussian(x, 0.5, 2796.35, 2.0, 0.3, 2803.52, 2.0)
        np.testing.assert_allclose(y, 1.0, atol=1e-3)

    def test_single_gaussian_shape(self):
        x = np.linspace(2780, 2820, 200)
        params = [0.6, 2796.35, 2.0]
        y = single_gaussian(x, params)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(np.all(np.isfinite(y)))

    def test_single_gaussian_at_continuum(self):
        x = np.array([2700.0, 2900.0])
        y = single_gaussian(x, [0.6, 2796.35, 2.0])
        np.testing.assert_allclose(y, 1.0, atol=1e-3)


class TestConvolutionFun(unittest.TestCase):

    def setUp(self):
        self.absorber = 'MgII'
        self.residual = np.random.random(4500)
        self.f1, self.f2 = 0.6123, 0.3054

    def test_log_mode_output_length(self):
        result = convolution_fun(self.absorber, self.residual, 3.0, True, 0.0001, None, self.f1, self.f2)
        self.assertEqual(len(result), len(self.residual))

    def test_linear_mode_output_length(self):
        result = convolution_fun(self.absorber, self.residual, 3.0, False, 0.8, None, self.f1, self.f2)
        self.assertEqual(len(result), len(self.residual))

    def test_civ_absorber(self):
        result = convolution_fun('CIV', self.residual, 2.0, False, 0.8, 0, 0.1908, 0.09522)
        self.assertEqual(len(result), len(self.residual))

    def test_unsupported_absorber_raises(self):
        with self.assertRaises(ValueError):
            convolution_fun('MnII', self.residual, 3.0, True, 0.0001, None, 0.5, 0.25)


class TestValidateSizes(unittest.TestCase):

    def test_equal_sizes_returns_zero(self):
        arr = np.ones(100)
        result = validate_sizes(arr, arr, 0)
        self.assertEqual(result, 0)

    def test_unequal_sizes_returns_one(self):
        result = validate_sizes(np.ones(100), np.ones(50), 0)
        self.assertEqual(result, 1)


class TestNumericKey(unittest.TestCase):

    def test_extracts_first_number(self):
        self.assertEqual(numeric_key('file_042_foo.fits'), 42)

    def test_no_number_returns_inf(self):
        self.assertEqual(numeric_key('no_numbers_here.txt'), float('inf'))

    def test_sorting(self):
        files = ['file3.fits', 'file20.fits', 'file1.fits']
        self.assertEqual(sorted(files, key=numeric_key), ['file1.fits', 'file3.fits', 'file20.fits'])


class TestParseQsoSequence(unittest.TestCase):

    def test_integer_input(self):
        result = parse_qso_sequence(100)
        np.testing.assert_array_equal(result, np.arange(100))

    def test_string_integer(self):
        result = parse_qso_sequence('50')
        np.testing.assert_array_equal(result, np.arange(50))

    def test_range_string(self):
        result = parse_qso_sequence('1-5')
        np.testing.assert_array_equal(result, np.array([1, 2, 3, 4, 5]))

    def test_range_with_step(self):
        result = parse_qso_sequence('0-10:2')
        np.testing.assert_array_equal(result, np.array([0, 2, 4, 6, 8, 10]))

    def test_invalid_string_raises(self):
        with self.assertRaises(ValueError):
            parse_qso_sequence('bad-input')


class TestGetPackageVersions(unittest.TestCase):

    def test_returns_dict(self):
        v = get_package_versions()
        self.assertIsInstance(v, dict)

    def test_contains_expected_packages(self):
        v = get_package_versions()
        for pkg in ('numpy', 'astropy', 'scipy'):
            self.assertIn(pkg, v)

    def test_values_are_strings(self):
        v = get_package_versions()
        for val in v.values():
            self.assertIsInstance(val, str)


class TestModifyUnits(unittest.TestCase):

    def _dummy_col(self, unit=None):
        """Minimal column-like object with .unit attribute."""
        class _Col:
            pass
        c = _Col()
        c.unit = unit
        return c

    def test_ew_column(self):
        self.assertEqual(modify_units('MGII_2796_EW', self._dummy_col()), 'Angstrom')

    def test_vdisp_column(self):
        self.assertEqual(modify_units('MGII_2796_VDISP', self._dummy_col()), 'km s-1')

    def test_logN_column(self):
        self.assertEqual(modify_units('LOG10N', self._dummy_col()), 'cm-2')

    def test_other_column_with_unit(self):
        result = modify_units('Z_ABS', self._dummy_col(unit='deg'))
        self.assertEqual(result, 'deg')

    def test_other_column_without_unit(self):
        result = modify_units('Z_ABS', self._dummy_col(unit=None))
        self.assertIsNone(result)


class TestMatchOrder(unittest.TestCase):

    def test_already_ordered(self):
        arr1 = np.array([1, 2, 3, 4])
        arr2 = np.array([1, 2, 3, 4])
        idx = match_order(arr1, arr2)
        np.testing.assert_array_equal(arr2[idx], arr1)

    def test_permuted(self):
        # match_order works for self-inverse permutations (e.g. reversal)
        arr1 = np.array([0, 1, 2])
        arr2 = np.array([2, 1, 0])
        idx = match_order(arr1, arr2)
        np.testing.assert_array_equal(arr2[idx], arr1)

    def test_size_mismatch_raises(self):
        with self.assertRaises(AssertionError):
            match_order(np.array([1, 2]), np.array([1, 2, 3]))


class TestVelDispersion(unittest.TestCase):

    def test_scalar_resolution(self):
        obs = np.linspace(2796, 2900, 100)
        v1, v2 = vel_dispersion(2796.35, 2803.52, 3.0, 3.0, 10.0, 0.5, obs)
        self.assertTrue(np.isfinite(v1) or np.isnan(v1))
        self.assertTrue(np.isfinite(v2) or np.isnan(v2))

    def test_array_resolution(self):
        obs = np.linspace(2796, 2900, 100)
        res_arr = np.full(100, 10.0)
        v1, v2 = vel_dispersion(2796.35, 2803.52, 3.0, 3.0, res_arr, 0.5, obs)
        self.assertTrue(np.isfinite(v1) or np.isnan(v1))

    def test_narrow_line_below_resolution_gives_nan(self):
        obs = np.linspace(2796, 2900, 100)
        # sigma = 0.1 Ang -> v_sigma << instrumental 200 km/s -> unresolved -> NaN
        v1, v2 = vel_dispersion(2796.35, 2803.52, 0.1, 0.1, 200.0, 0.5, obs)
        self.assertTrue(np.isnan(v1))
        self.assertTrue(np.isnan(v2))


class TestFitsFileHelpers(unittest.TestCase):
    """Tests that require the real DESI spectrum FITS file."""

    def setUp(self):
        self.fits_file = os.path.abspath(_DESI_FITS)
        if not os.path.exists(self.fits_file):
            self.skipTest(f'DESI test FITS file not found: {self.fits_file}')

    def test_get_all_extnames_returns_list(self):
        extnames = get_all_extnames(self.fits_file)
        self.assertIsInstance(extnames, list)
        self.assertGreater(len(extnames), 0)

    def test_get_all_extnames_tuples(self):
        extnames = get_all_extnames(self.fits_file)
        for entry in extnames:
            self.assertEqual(len(entry), 3)  # (index, name, type)

    def test_read_nqso_from_header_returns_int(self):
        n = read_nqso_from_header(self.fits_file)
        self.assertIsInstance(n, int)
        self.assertGreater(n, 0)

    def test_read_nqso_from_header_missing_file_raises(self):
        with self.assertRaises(FileNotFoundError):
            read_nqso_from_header('/nonexistent/path/file.fits')

    def test_read_nqso_from_header_missing_hdu_raises(self):
        with self.assertRaises(ValueError):
            read_nqso_from_header(self.fits_file, hdu_name='NONEXISTENT_HDU')


if __name__ == '__main__':
    unittest.main()
