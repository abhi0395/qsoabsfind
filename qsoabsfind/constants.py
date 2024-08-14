"""
constants.py

This script contains constants and dictionaries used in the QSO spectra analysis.
"""

speed_of_light = 3e5  # Speed of light in km/s
#dict: Dictionary containing line data.
lines = {
    'MgII_2796': 2796.35,
    'MgII_2803': 2803.52,
    'MgI_2799': 2799.117,
    'dz_start': 0.018,
    'dz_end': 0.003,
    'CIV_1548': 1548.20,
    'CIV_1550': 1550.77,
    'CIV_1549': 1549.48,
    'dv': -5000,
    'Lya': 1215.16
}

# dict: Dictionary containing default parameters for absorber search.

ker_width_pixels = [3, 4, 5, 6, 7, 8]  # Pixel size for Gaussian convolution
pm_pixel = 200  # Pixel size for error calculation to define the threshold for potential absorber features
mult_resi = 1  # Coefficients to multiply the residual before Gaussian search
lam_sep = 300 # wavelength shift from edges (in Ang), i.e., lam_min + lam_sep, lam_max - lam_sep, to avoid noisy edges of the spectrum

# Some NOTES:
# SDSS spectra: use resolution 69 (km/s), wave_res = 0.0001, logwave=True
# DESI spectra: use resolution None, wave_res = 0.8, logwave=False, average resolution will be calculated by code

search_parameters = {
    'MgII': {
        'ker_width_pixels': ker_width_pixels,
        'pm_pixel': pm_pixel,
        'coeff_sigma': 2.5,
        'mult_resi': mult_resi,
        'd_pix': 0.6,
        'sn_line1':3,
        'sn_line2':2,
        'use_covariance':False,
        'resolution':69,
        'logwave':True,
        'wave_res':0.0001,
        'lam_edge_sep':lam_sep,
        'verbose':False,
    },
    'CIV': {
        'ker_width_pixels': ker_width_pixels,
        'pm_pixel': pm_pixel,
        'coeff_sigma': 2,
        'mult_resi': mult_resi,
        'd_pix': 0.7,
        'sn_line1':3,
        'sn_line2':2,
        'use_covariance':False,
        'resolution':69,
        'logwave':True,
        'wave_res':0.0001,
        'lam_edge_sep':lam_sep,
        'verbose':False,
    }
}

# Default amplitudes for different absorbers
amplitude_dict = {
    'MgII': 0.94,
    'CIV': 0.75,  # Example value, you can change it accordingly
    'FeII': 0.75  # Example value, you can change it accordingly
}

oscillator_parameters = {'MgII_f1':0.6123, 'MgII_f2':0.3954,
                        'CIV_f1':0.19, 'CIV_f2':0.0962

}
