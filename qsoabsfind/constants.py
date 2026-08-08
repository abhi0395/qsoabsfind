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

LARGE_WAVE = 1e6  # observed-frame hard upper wavelength limit (Ang); default 1e6 = no restriction
SMALL_WAVE = 100.0   # observed-frame hard lower wavelength limit (Ang); default 100 = no restriction
MIN_NPIXEL = 100 # minimum number of pixels required for a valid search region
LAM_CIV_MIN = 1310.0 # minimum wavelength to search for CIV absorbers (to avoid confusion with Silicon forest)

# ==============================
# Gaussian fit bound parameters
# ==============================

GAUSS_FIT_BD_CT = 2.0   # line-centre bound multiplier: allowed centre shift = BD_CT * d_pix
GAUSS_FIT_X_SEP = 30    # line-width upper cap: max sigma = X_SEP * del_sigma
GAUSS_FIT_EDGE  = 0.1   # small numerical buffer on sigma bounds to avoid hard boundary at zero
FIT_PARAM_SNR   = 1.0   # minimum fit-parameter SNR: each param must exceed its fitted error

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

# =============================
# Major emission lines in QSOs
# =============================

QSO_EMISSION_LINES = {
            "LyA": 1215.67,
            "NV": 1240.81,
            "SiIV_OIV": 1400.0,
            "CIV": 1549.0,
            "CIII": 1908.7,
            "MgII": 2798.0,
            "OII_3727": 3727.0,
            "Hbeta": 4861.33,
            "OIII": 5007.0,
            "Halpha": 6562.8,
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
# Some Algorithmic parameters
# (Do not affect the overall search of absorbers,
# but can be changed if user wants to explore.
# Though I would suggest to not change much)
# ==============================

MIN_PIXELS_PER_PARAM = 2         # minimum pixels per free parameter required to constrain the double-Gaussian fit (min_pixels = MIN_PIXELS_PER_PARAM * nparam)
CANDIDATE_VALIDATION_NPIX = 3   # pixels around a line minimum for candidate validation in find_valid_indices
SNR_DEFAULT_DPIX = 5             # fallback pixel window for SNR estimation when Gaussian sigma is unavailable
SNR_NSIG = 3                     # Gaussian sigma multiplier for SNR integration window (~99.7% of flux)
REDSHIFT_REFINE_WINDOW = 9       # pixel window for refining redshift by locating the flux minimum
MEDIAN_WEIGHT_GAMMA = 4          # power-law exponent for 1/lambda^gamma weighting in median_selection_after_combining
CANDIDATE_DEDUP_CT = 2           # pixel tolerance multiplier for deduplicating close candidates
MAX_VEL_DISPERSION = 50         # maximum allowed velocity difference between doublet components (km/s)
CONV_KERNEL_EXTENT = 10          # convolution kernel half-extent: +/-N x sigma from line centre
FIT_WINDOW_HALF_WIDTH = 15       # Gaussian fitting window half-width multiplier: d_pix * N Ang on each side
GAUSS_FIT_NUM_ITER = 500         # maximum curve_fit iterations for double-Gaussian fitting
GAUSS_FIT_FINAL_ITER_FACTOR = 2  # multiplier applied to num_iter for the final fitting pass
GAUSS_FIT_BOOT_ITER_FACTOR = 0.4 # fraction of num_iter used per bootstrap fit
GAUSS_FIT_BOOT_WARM_SPREAD = 0.1 # fractional std for initial perturbation of best-fit params in bootstrap
GAUSS_FIT_FTOL = 1e-4            # function convergence tolerance for scipy curve_fit
GAUSS_FIT_XTOL = 1e-4            # parameter convergence tolerance for scipy curve_fit
GAUSS_AMP_MIN = 0.025             # minimum amplitude floor for Gaussian initial conditions
GAUSS_AMP_MAX = 0.975             # maximum amplitude cap for Gaussian initial conditions
GAUSS_SIGMA_INIT_MIN = 0.2       # lower bound (Ang) for sigma draw when no bounds are supplied
GAUSS_SIGMA_INIT_MAX = 5.0       # upper bound (Ang) for sigma draw when no bounds are supplied
SIGNIFICANCE_N_PIXELS = 2        # pixel window around each line centre for the absorption check in quick_significance_test
EW_FIT_WINDOW = 5                # pixel window for redshift refinement during EW measurement
AODM_FLUX_CLIP_MIN = 0.005       # minimum flux clipped before log computation in AODM to avoid log(0)
AODM_INCONSISTENT_SIGMA = 2.0  # sigma threshold for flagging inconsistent AODM measurements
ZABS_KNOWN_MAX_DV = 500          # maximum allowed velocity offset (km/s) between fitted and seed redshift in known-z mode

# ==============================
# User-overridable parameters
# All scalar parameters listed here can be overridden by a user-provided constants file.
# True physical constants (speed_of_light, lines, oscillator_parameters) and
# dict-type absorber registries (doublet_keys, amplitude_dict) are excluded.
# ==============================
OVERRIDABLE_CONSTANTS = (

    # Instrument / survey wavelength limits and search thresholds

    'SMALL_WAVE',
    'LARGE_WAVE',
    'MIN_NPIXEL',
    'LAM_CIV_MIN',

    # Gaussian fit bound parameters

    'GAUSS_FIT_BD_CT',
    'GAUSS_FIT_X_SEP',
    'GAUSS_FIT_EDGE',
    'FIT_PARAM_SNR',

    # Algorithmic parameters

    'MIN_PIXELS_PER_PARAM',
    'CANDIDATE_VALIDATION_NPIX',
    'SNR_DEFAULT_DPIX',
    'SNR_NSIG',
    'REDSHIFT_REFINE_WINDOW',
    'MEDIAN_WEIGHT_GAMMA',
    'CANDIDATE_DEDUP_CT',
    'MAX_VEL_DISPERSION',
    'CONV_KERNEL_EXTENT',
    'FIT_WINDOW_HALF_WIDTH',
    'GAUSS_FIT_NUM_ITER',
    'GAUSS_FIT_FINAL_ITER_FACTOR',
    'GAUSS_FIT_BOOT_ITER_FACTOR',
    'GAUSS_FIT_BOOT_WARM_SPREAD',
    'GAUSS_FIT_FTOL',
    'GAUSS_FIT_XTOL',
    'GAUSS_AMP_MIN',
    'GAUSS_AMP_MAX',
    'GAUSS_SIGMA_INIT_MIN',
    'GAUSS_SIGMA_INIT_MAX',
    'SIGNIFICANCE_N_PIXELS',
    'EW_FIT_WINDOW',
    'AODM_FLUX_CLIP_MIN',
    'AODM_INCONSISTENT_SIGMA',
    'ZABS_KNOWN_MAX_DV',
    'QSO_EMISSION_LINES',
)

# ==============================
# Notes:
# ==============================

"""
- SDSS spectra: `logwave=True` (log-uniform wavelength); resolution estimated automatically.
- DESI spectra: `logwave=False` (linear wavelength grid); per-pixel resolution is computed.
  Adjust as needed based on spectrum type.
"""
