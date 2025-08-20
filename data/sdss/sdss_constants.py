"""
sdss_constants.py

This module defines search parameters. It includes general and survey-specific configurations
for SDSS quasar spectra.

Used in:
- data/deso/qso_test_spectra.py
- To generate: data/sdss/MgII_cat.fits and data/sdss/CIV_cat.fits

Usage:
    from constants import speed_of_light, lines, search_parameters, ...
"""

# ==============================
# Search Parameter Dictionary
# ==============================

search_parameters = {
    'ker_width_pixels':[3, 4, 5, 6, 7, 8], # Gaussian kernel widths (in pixels) for convolution
    'pm_pixel': 200, # Window size around feature for threshold calculation for convolved array
    'coeff_sigma': 2, # for convolution the SNR cut
    'mult_resi': 1, # Multiplication factor for residual spectrum (to shift the normalization up or down)
    'd_pix': 0.6, # tolerance (in Angs) for line difference from true values
    'sn_line1': 3, # SNR cut for first line
    'sn_line2': 2, # SNR cut for second line
    'use_covariance': False, # to use covariance matrix in EW error calculation
    'logwave': True,  # Assume SDSS-style log scale by default
    'lam_edge_sep': 50, # Wavelength cut from spectrum edges (in Ang)
    'conf_level':0.95, # 95 percent confidence level for absorber selection
    'verbose': True, # for printing statements for debugging
    'dv': 5000,  # velocity offset from quasars redshift in km/s
    'start_rest_wave':None, # blue end of rest-frame quasar wavelength, None --> default
    'end_rest_wave':None, # red end of rest-frame quasar wavelength, None --> default
    'nboot':None, # if provided the number, will perform bootstrapping for EW error calculation
    'continuum_error_frac':0.05 # Fixed fractional error due to continuum normalization (5%) used AODM column density measurements
}

# ==============================
# Notes:
# ==============================

"""
- SDSS spectra: `logwave=False` (linear wavelength grid); per-pixel resolution is computed.
  - MgII/CIV default parameters are optimized for SDSS spectra
  Adjust as needed based on spectrum type.

  Fixed fractional error due to continuum normalization (5%)
  Used in estimating the uncertainty on column density
  arising from continuum placement uncertainties.
  Currently set empirically — a more optimal estimate can be obtained
  by stacking continuum-normalized residual spectra in the observed frame
  and measuring the standard deviation from unity.
"""
