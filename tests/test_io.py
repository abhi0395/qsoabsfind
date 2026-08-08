"""
Tests for io.py -- read_fits_file, save_results_to_fits, append_table_to_fits,
read_any_fits_file.
Uses the real SDSS and DESI test FITS files so no synthetic data is needed.
"""

import os
import tempfile
import unittest
import numpy as np
from astropy.table import Table
from astropy.io import fits

from qsoabsfind.io import (
    read_fits_file,
    save_results_to_fits,
    append_table_to_fits,
    read_any_fits_file,
)

_SDSS_FITS = os.path.join(os.path.dirname(__file__), '..', 'data', 'sdss', 'qso_test_spectra.fits')
_DESI_FITS = os.path.join(os.path.dirname(__file__), '..', 'data', 'desi', 'qso_test_spectra.fits')


class TestReadFitsFile(unittest.TestCase):

    def setUp(self):
        self.fits_file = os.path.abspath(_DESI_FITS)
        if not os.path.exists(self.fits_file):
            self.skipTest(f'DESI test FITS file not found: {self.fits_file}')

    # --- read all spectra (index=None) ---

    def test_returns_five_items(self):
        result = read_fits_file(self.fits_file)
        self.assertEqual(len(result), 5)

    def test_header_is_fits_header(self):
        header, *_ = read_fits_file(self.fits_file)
        self.assertIsInstance(header, fits.Header)

    def test_wavelength_1d(self):
        _, _, _, wavelength, _ = read_fits_file(self.fits_file)
        self.assertEqual(wavelength.ndim, 1)

    def test_flux_2d_when_no_index(self):
        _, flux, _, _, _ = read_fits_file(self.fits_file)
        self.assertEqual(flux.ndim, 2)

    def test_metadata_is_table(self):
        _, _, _, _, meta = read_fits_file(self.fits_file)
        self.assertIsInstance(meta, Table)

    # --- read single spectrum (int index) ---

    def test_flux_1d_for_int_index(self):
        _, flux, error, _, _ = read_fits_file(self.fits_file, index=0)
        self.assertEqual(flux.ndim, 1)
        self.assertEqual(error.ndim, 1)

    def test_flux_length_matches_wavelength_for_int_index(self):
        _, flux, _, wavelength, _ = read_fits_file(self.fits_file, index=0)
        self.assertEqual(flux.size, wavelength.size)

    def test_metadata_type_for_int_index(self):
        # For int index the metadata is a single astropy Row
        from astropy.table import Row
        _, _, _, _, meta = read_fits_file(self.fits_file, index=0)
        self.assertIsInstance(meta, Row)

    # --- read multiple spectra (list/array index) ---

    def test_flux_2d_for_array_index(self):
        idx = np.array([0, 1, 2])
        _, flux, error, _, _ = read_fits_file(self.fits_file, index=idx)
        self.assertEqual(flux.ndim, 2)
        self.assertEqual(flux.shape[0], 3)

    def test_metadata_length_matches_index_length(self):
        idx = [0, 1, 2, 3]
        _, _, _, _, meta = read_fits_file(self.fits_file, index=idx)
        self.assertEqual(len(meta), 4)


class TestReadFitsFileSDSS(unittest.TestCase):
    """Same basic checks with the SDSS file."""

    def setUp(self):
        self.fits_file = os.path.abspath(_SDSS_FITS)
        if not os.path.exists(self.fits_file):
            self.skipTest(f'SDSS test FITS file not found: {self.fits_file}')

    def test_sdss_wavelength_1d(self):
        _, _, _, wavelength, _ = read_fits_file(self.fits_file)
        self.assertEqual(wavelength.ndim, 1)

    def test_sdss_single_index(self):
        _, flux, error, wavelength, _ = read_fits_file(self.fits_file, index=5)
        self.assertEqual(flux.size, wavelength.size)
        self.assertEqual(error.size, wavelength.size)


class TestReadAnyFitsFile(unittest.TestCase):

    def setUp(self):
        self.fits_file = os.path.abspath(_DESI_FITS)
        if not os.path.exists(self.fits_file):
            self.skipTest(f'DESI test FITS file not found: {self.fits_file}')

    def test_read_metadata_hdu_returns_table(self):
        hdr, data = read_any_fits_file(self.fits_file, 'METADATA')
        self.assertIsInstance(data, Table)
        self.assertIsInstance(hdr, fits.Header)

    def test_read_flux_hdu_returns_array(self):
        hdr, data = read_any_fits_file(self.fits_file, 'FLUX')
        self.assertIsNotNone(data)
        self.assertTrue(hasattr(data, 'shape'))

    def test_read_wavelength_hdu_returns_array(self):
        _, data = read_any_fits_file(self.fits_file, 'WAVELENGTH')
        self.assertEqual(data.ndim, 1)


class TestSaveResultsToFits(unittest.TestCase):

    def _make_results(self, n=3):
        return {
            'index_spec': list(range(n)),
            'z_abs': [0.5 + 0.1 * i for i in range(n)],
            'gauss_fit': [np.ones(6) * (i + 0.1) for i in range(n)],
            'gauss_fit_std': [np.ones(6) * 0.01 for _ in range(n)],
            'ew_1_mean': [0.5] * n,
            'ew_2_mean': [0.3] * n,
            'ew_total_mean': [0.8] * n,
            'ew_1_error': [0.05] * n,
            'ew_2_error': [0.04] * n,
            'ew_total_error': [0.06] * n,
            'z_abs_err': [0.001] * n,
            'sn_1': [5.0] * n,
            'sn_2': [4.0] * n,
            'vel_disp1': [30.0] * n,
            'vel_disp2': [28.0] * n,
            'vel_disp1_err': [3.0] * n,
            'vel_disp2_err': [2.8] * n,
            'delta_chi2_line1': [20.0] * n,
            'delta_chi2_line2': [18.0] * n,
            'pure_redchi2': [15.0] * n,
        }

    def setUp(self):
        self.input_file = os.path.abspath(_DESI_FITS)
        if not os.path.exists(self.input_file):
            self.skipTest(f'DESI test FITS file not found: {self.input_file}')
        self.headers = {
            'SURVEY': {'value': 'DESI', 'comment': 'Survey name'},
        }

    def test_creates_output_file(self):
        results = self._make_results()
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            out = f.name
        try:
            save_results_to_fits(results, self.input_file, out, self.headers, 'MgII')
            self.assertTrue(os.path.exists(out))
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_output_has_absorber_hdu(self):
        results = self._make_results()
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            out = f.name
        try:
            save_results_to_fits(results, self.input_file, out, self.headers, 'MgII')
            with fits.open(out) as hdul:
                names = [h.name for h in hdul]
            self.assertIn('ABSORBER', names)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_output_has_metadata_hdu(self):
        results = self._make_results()
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            out = f.name
        try:
            save_results_to_fits(results, self.input_file, out, self.headers, 'MgII')
            with fits.open(out) as hdul:
                names = [h.name for h in hdul]
            self.assertIn('METADATA', names)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_absorber_row_count_matches_results(self):
        results = self._make_results(n=3)
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            out = f.name
        try:
            save_results_to_fits(results, self.input_file, out, self.headers, 'MgII')
            t = Table.read(out, hdu='ABSORBER')
            self.assertEqual(len(t), 3)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_civ_absorber_columns(self):
        results = self._make_results(n=2)
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            out = f.name
        try:
            save_results_to_fits(results, self.input_file, out, self.headers, 'CIV')
            t = Table.read(out, hdu='ABSORBER')
            self.assertIn('CIV_1548_EW', t.colnames)
            self.assertIn('CIV_1548_VDISP_ERR', t.colnames)
        finally:
            if os.path.exists(out):
                os.remove(out)

    def test_unsupported_absorber_raises(self):
        results = self._make_results()
        with tempfile.NamedTemporaryFile(suffix='.fits', delete=False) as f:
            out = f.name
        try:
            with self.assertRaises(ValueError):
                save_results_to_fits(results, self.input_file, out, self.headers, 'Vibranium')
        finally:
            if os.path.exists(out):
                os.remove(out)


class TestAppendTableToFits(unittest.TestCase):

    def setUp(self):
        self.input_file = os.path.abspath(_DESI_FITS)
        if not os.path.exists(self.input_file):
            self.skipTest(f'DESI test FITS file not found: {self.input_file}')

    def _make_output_file(self, headers=None):
        """Create a minimal FITS file for appending tests."""
        results = {
            'index_spec': [0], 'z_abs': [0.5], 'gauss_fit': [np.ones(6)],
            'gauss_fit_std': [np.ones(6) * 0.01], 'ew_1_mean': [0.5],
            'ew_2_mean': [0.3], 'ew_total_mean': [0.8], 'ew_1_error': [0.05],
            'ew_2_error': [0.04], 'ew_total_error': [0.06], 'z_abs_err': [0.001],
            'sn_1': [5.0], 'sn_2': [4.0], 'vel_disp1': [30.0], 'vel_disp2': [28.0], 'vel_disp1_err': [1.5], 'vel_disp2_err':[1.5],
            'delta_chi2_line1': [20.0], 'delta_chi2_line2': [18.0],
            'pure_redchi2': [15.0],
        }
        hdrs = headers or {'SURVEY': {'value': 'DESI', 'comment': ''}}
        f = tempfile.NamedTemporaryFile(suffix='.fits', delete=False)
        out = f.name
        f.close()
        save_results_to_fits(results, self.input_file, out, hdrs, 'MgII')
        return out

    def test_appends_new_hdu(self):
        out = self._make_output_file()
        try:
            extra = Table({'LOG10N': [12.5], 'SIG_LOG10N': [0.1]})
            append_table_to_fits(out, extra, 'COLUMN_DENSITY')
            with fits.open(out) as hdul:
                names = [h.name for h in hdul]
            self.assertIn('COLUMN_DENSITY', names)
        finally:
            os.remove(out)

    def test_appended_hdu_has_correct_data(self):
        out = self._make_output_file()
        try:
            extra = Table({'LOG10N': [12.5, 13.0], 'SIG_LOG10N': [0.1, 0.2]})
            append_table_to_fits(out, extra, 'COLDENS')
            t = Table.read(out, hdu='COLDENS')
            self.assertEqual(len(t), 2)
            self.assertAlmostEqual(float(t['LOG10N'][0]), 12.5)
        finally:
            os.remove(out)

    def test_non_table_raises_type_error(self):
        out = self._make_output_file()
        try:
            with self.assertRaises(TypeError):
                append_table_to_fits(out, {'not': 'a table'}, 'BAD')
        finally:
            os.remove(out)

    def test_missing_file_raises_value_error(self):
        extra = Table({'X': [1]})
        with self.assertRaises(ValueError):
            append_table_to_fits('/nonexistent/path.fits', extra, 'TEST')


if __name__ == '__main__':
    unittest.main()
