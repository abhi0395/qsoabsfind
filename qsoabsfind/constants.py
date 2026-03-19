"""
constants.py

This module defines constants, line wavelengths, search parameters, and oscillator strengths
used in QSO absorption line analysis.

Usage:
    from constants import speed_of_light, lines, doublet_keys ...
"""

# ==============================
# Physical Constants
# ==============================

speed_of_light = 299792.458   # Speed of light in km/s

LARGE_WAVE = 9800.0 # for SDSS/DESI observed wave
SMALL_WAVE = 500.0 # for SDSS/DESI observed wave

MIN_NPIXEL = 100 # minimum number of pixels required for a valid search region

LAM_CIV_MIN = 1310.0 # minimum wavelength to search for CIV absorbers (to avoid confusion with Silicon forest)

# ===================
# Supported absorbers
# ===================

doublet_keys = {
        'MgII': ('MgII_2796', 'MgII_2803'),
        'CIV':  ('CIV_1548', 'CIV_1550'),
        'OVI':  ('OVI_1032', 'OVI_1038'),
        'NV':   ('NV_1238', 'NV_1242'),
        'SiIV': ('SiIV_1394', 'SiIV_1403'),
        'AlIII': ('AlIII_1855', 'AlIII_1863'),
        'FeII': ('FeII_2586', 'FeII_2600'),
        'CaII': ('CaII_3934', 'CaII_3969'),
        'NaI': ('NaI_5891', 'NaI_5897')
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

    # OVI doublet
    'OVI_1032': 1031.926,
    'OVI_1038': 1037.617,
    'OVI_1033': 1033.82,  # OVI emission from Quasar

    # NV doublet
    'NV_1238': 1238.821,
    'NV_1242': 1242.804,
    'NV_1240': 1240.81,  # NV emission from Quasar

    # SiIV doublet
    'SiIV_1394': 1393.755,
    'SiIV_1403': 1402.770,
    'SiIV_1399': 1399.8,  # SiIV emission from Quasar

    # AlIII doublet
    'AlIII_1855': 1854.716,
    'AlIII_1863': 1862.790,
    'AlIII_1857': 1857.4,  # AlIII emission from Quasar

    # FeII lines (two strongest)
    'FeII_2586': 2586.650,
    'FeII_2600': 2600.173,

    # CaII doublet
    'CaII_3934': 3934.78,
    'CaII_3969': 3969.59,

    # NaI doublet
    'NaI_5891': 5891.583,
    'NaI_5897': 5897.566
}

# ==============================
# Default Amplitudes for Gaussian doublet kernel
# ==============================

amplitude_dict = {
    'MgII': 0.94,
    'CIV': 0.75,
    'FeII': 0.75,
    'AlIII': 0.5,
    'SiIV': 0.5,
    'OVI': 0.5,
    'NV': 0.5,
    'NaI': 0.5,
    'CaII': 0.5

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
    'FeII_f2': 0.239,    # 2600.173

    # CaII doublet
    'CaII_f1': 0.6346,     # 3934.78
    'CaII_f2': 0.3116,     # 3969.59

    # NaI doublet
    'NaI_f1': 0.6405,     # 5891.583
    'NaI_f2': 0.3199      # 5897.566
}


# ==============================
# Notes:
# ==============================

"""
- SDSS spectra: `logwave=True` (log-uniform wavelength); resolution estimated automatically.
- DESI spectra: `logwave=False` (linear wavelength grid); per-pixel resolution is computed.
  Adjust as needed based on spectrum type.
"""
