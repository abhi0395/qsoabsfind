"""
sdss_constants.py

This module defines constants, line wavelengths, search parameters, and oscillator strengths
used in QSO absorption line analysis. It includes general and survey-specific configurations
for SDSS quasar spectra.

Used in:
- data/sdss/qso_test_spectra.py
- To generate: data/sdss/MgII_cat.fits and data/sdss/CIV_cat.fits

Usage:
    from constants import speed_of_light, lines, search_parameters, ...
"""

# ==============================
# Physical Constants
# ==============================

speed_of_light = 3e5  # Speed of light in km/s

continuum_systematic_error = 0.05
# Fixed fractional error due to continuum normalization (5%)
# Used in estimating the uncertainty on column density
# arising from continuum placement uncertainties.
# Currently set empirically — a more optimal estimate can be obtained
# by stacking continuum-normalized residual spectra in the observed frame
# and measuring the standard deviation from unity.

# supported absorbers
doublet_keys = {
        'MgII': ('MgII_2796', 'MgII_2803'),
        'CIV':  ('CIV_1548', 'CIV_1550'),
    }

# ==============================
# Absorption Line Wavelengths (in Ang)
# ==============================

lines = {
    # Lyman series
    'Lya': 1215.67,
    'Lyb_1026': 1025.72,

    # CIV doublet
    'CIV_1548': 1548.20,
    'CIV_1550': 1550.77,
    'CIV_1549': 1549.48,  # CIV emission from quasar

    # MgII doublet
    'MgII_2796': 2796.35,
    'MgII_2803': 2803.52,
    'MgII_2799': 2799.117,  # MgII emission from quasar

    'dv': 5000,  # velocity offset from quasars redshift in km/s
    'start_rest_wave':None, # blue end of rest-frame quasar wavelength, None --> default
    'end_rest_wave':None # red end of rest-frame quasar wavelength, None --> default
}


# ==============================
# Default Signal Parameters
# ==============================

ker_width_pixels = [3, 4, 5, 6, 7, 8]  # Gaussian kernel widths (in pixels) for convolution
pm_pixel = 200                        # Window size around feature for threshold calculation for convolved array
mult_resi = 1                         # Multiplication factor for residual spectrum
lam_sep = 25                         # Wavelength cut from spectrum edges (in Ang)

# ==============================
# Search Parameter Dictionary
# ==============================

search_parameters = {
    'MgII': {
        'ker_width_pixels': ker_width_pixels,
        'pm_pixel': pm_pixel,
        'coeff_sigma': 2.5,
        'mult_resi': mult_resi,
        'd_pix': 0.6,
        'sn_line1': 3,
        'sn_line2': 2,
        'use_covariance': False,
        'logwave': True,  # SDSS-like log-scaled wavelength
        'lam_edge_sep': lam_sep,
        'verbose': True,
    },
    'CIV': {
        'ker_width_pixels': ker_width_pixels,
        'pm_pixel': pm_pixel,
        'coeff_sigma': 2,
        'mult_resi': mult_resi,
        'd_pix': 0.6,
        'sn_line1': 3,
        'sn_line2': 2,
        'use_covariance': False,
        'logwave': True,  # SDSS-like log-scaled wavelength
        'lam_edge_sep': lam_sep,
        'verbose': True,
    }
}

# ==============================
# Default Amplitudes for Gaussian doublet kernel
# ==============================

amplitude_dict = {
    'MgII': 0.94,
    'CIV': 0.75,
}

# ==============================
# Oscillator Strengths (f-values)
# ==============================

oscillator_parameters = {
    'MgII_f1': 0.6123,  # MgII 2796
    'MgII_f2': 0.3954,  # MgII 2803
    'CIV_f1': 0.19,     # CIV 1548
    'CIV_f2': 0.0962    # CIV 1550
}

# ==============================
# Notes:
# ==============================

"""
- SDSS spectra: `logwave=True` (log-uniform wavelength); resolution estimated automatically.
- MgII/CIV default parameters are optimized for SDSS spectra
  Adjust as needed based on spectrum type.
"""
