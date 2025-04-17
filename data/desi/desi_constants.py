"""
desi_constants.py

This module defines constants, line wavelengths, search parameters, and oscillator strengths
used in QSO absorption line analysis. It includes general and survey-specific configurations
for DESI quasar spectra.

Used in:
- data/deso/qso_test_spectra.py
- To generate: data/desi/MgII_cat.fits and data/desi/CIV_cat.fits

Usage:
    from constants import speed_of_light, lines, search_parameters, ...
"""

# ==============================
# Physical Constants
# ==============================

speed_of_light = 3e5  # Speed of light in km/s

# ==============================
# Absorption Line Wavelengths (in Ang)
# ==============================

lines = {
    'Lya': 1215.16,
    'MgII_2796': 2796.35,
    'MgII_2803': 2803.52,
    'MgI_2799': 2799.117, # average of the MgII doublet
    'CIV_1548': 1548.20,
    'CIV_1550': 1550.77,
    'CIV_1549': 1549.48,  # average of the CIV doublet
    'dz_start': 0.018,  # starting redshift offset for defining absorber window
    'dz_end': 0.003,    # ending redshift offset for defining absorber window
    'dv': -5000         # velocity offset from quasars redshift in km/s
}

# ==============================
# Default Signal Parameters
# ==============================

ker_width_pixels = [3, 4, 5, 6, 7, 8]  # Gaussian kernel widths (in pixels) for convolution
pm_pixel = 200                        # Window size around feature for threshold calculation for convolved array
mult_resi = 1                         # Multiplication factor for residual spectrum
lam_sep = 300                         # Wavelength cut from spectrum edges (Ang)

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
        'logwave': False,  # DESI-like log-scaled wavelength
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
        'logwave': False,  # DESI-like linear wavelength
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
    'FeII': 0.75
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
- DESI spectra: `logwave=False` (linear wavelength grid); per-pixel resolution is computed.
  - MgII/CIV default parameters are optimized for DESI spectra
  Adjust as needed based on spectrum type.
"""
