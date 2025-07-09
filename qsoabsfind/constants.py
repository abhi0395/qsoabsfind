"""
constants.py

This module defines constants, line wavelengths, search parameters, and oscillator strengths
used in QSO absorption line analysis.

Usage:
    from constants import speed_of_light, lines, search_parameters, ...
"""

# ==============================
# Physical Constants
# ==============================

speed_of_light = 3e5  # Speed of light in km/s

# supported absorbers

doublet_keys = {
        'MgII': ('MgII_2796', 'MgII_2803'),
        'CIV':  ('CIV_1548', 'CIV_1550'),
        'OVI':  ('OVI_1032', 'OVI_1038'),
        'NV':   ('NV_1238', 'NV_1242'),
        'SiIV': ('SiIV_1394', 'SiIV_1403'),
        'AlIII': ('AlIII_1855', 'AlIII_1863'),
        'FeII': ('FeII_2586', 'FeII_2600')
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
    'CIV_1549': 1549.48,  # average of the CIV doublet

    # MgII doublet
    'MgII_2796': 2796.35,
    'MgII_2803': 2803.52,
    'MgI_2799': 2799.117,  # average of the MgII doublet

    # OVI doublet
    'OVI_1032': 1031.926,
    'OVI_1038': 1037.617,
    'OVI_1035': 1034.77,  # average

    # NV doublet
    'NV_1238': 1238.821,
    'NV_1242': 1242.804,
    'NV_1240': 1240.81,  # average

    # SiIV doublet
    'SiIV_1394': 1393.755,
    'SiIV_1403': 1402.770,
    'SiIV_1399': 1398.26,  # average

    # AlIII doublet
    'AlIII_1855': 1854.716,
    'AlIII_1863': 1862.790,
    'AlIII_1857': 1858.753,  # average

    # FeII lines (two strongest)
    'FeII_2586': 2586.650,
    'FeII_2600': 2600.173,
    'FeII_2593': 2593.4115,  # average of above

    'dv': 5000  # velocity offset quasars redshift in km/s
}


# ==============================
# Default Signal Parameters
# ==============================

ker_width_pixels = [3, 4, 5, 6, 7, 8]  # Gaussian kernel widths (in pixels) for convolution
pm_pixel = 200                        # Window size around feature for threshold calculation for convolved array
mult_resi = 1                         # Multiplication factor for residual spectrum
lam_sep = 50                         # Wavelength cut from spectrum edges (in Ang)

# ==============================
# Search Parameter Dictionary
# ==============================

# Define default parameters
# Used in DESI-like spectra (e.g., data/desi/qso_test_spectra.py) for unittest
default_search_params = {
    'ker_width_pixels': ker_width_pixels,
    'pm_pixel': pm_pixel,
    'coeff_sigma': 2.5,
    'mult_resi': mult_resi,
    'd_pix': 0.6,
    'sn_line1': 3,
    'sn_line2': 2,
    'use_covariance': False,
    'logwave': True,  # Assume SDSS-style linear by default
    'lam_edge_sep': lam_sep,
    'verbose': True,
}

# Create the final dictionary
search_parameters = {
    absorber: default_search_params.copy()
    for absorber in doublet_keys.keys()
}

# ==============================
# Default Amplitudes for Gaussian doublet kernel
# ==============================

amplitude_dict = {
    'MgII': 0.94,
    'CIV': 0.75,
    'FeII': 0.75,
    'AlIII': 0.75,
    'SiIV': 0.75,
    'OVI': 0.75,
    'NV': 0.75,

}

# ==============================
# Oscillator Strengths (f-values)
# ==============================

oscillator_parameters = {
    # MgII doublet
    'MgII_f1': 0.6123,   # 2796.35
    'MgII_f2': 0.3054,   # 2803.52

    # CIV doublet
    'CIV_f1': 0.1908,    # 1548.20
    'CIV_f2': 0.09522,   # 1550.77

    # OVI doublet
    'OVI_f1': 0.1329,    # 1031.926
    'OVI_f2': 0.0661,    # 1037.617

    # NV doublet
    'NV_f1': 0.157,      # 1238.821
    'NV_f2': 0.07821,    # 1242.804

    # SiIV doublet
    'SiIV_f1': 0.514,    # 1393.755
    'SiIV_f2': 0.2553,   # 1402.770

    # AlIII doublet
    'AlIII_f1': 0.559,   # 1854.716
    'AlIII_f2': 0.278,   # 1862.790

    # FeII lines
    'FeII_f1': 0.0691,   # 2586.650
    'FeII_f2': 0.239    # 2600.173
}


# ==============================
# Notes:
# ==============================

"""
- SDSS spectra: `logwave=True` (log-uniform wavelength); resolution estimated automatically.
- DESI spectra: `logwave=False` (linear wavelength grid); per-pixel resolution is computed.
- MgII default parameters are optimized for SDSS spectra; CIV for DESI.
  Adjust as needed based on spectrum type.
"""
