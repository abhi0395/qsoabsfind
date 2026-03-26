Parameter File
==============

Constant File (Mandatory)
-------------------------

The user-defined **constants file** must follow the same structure as the `qsoabsfind.constants <https://github.com/abhi0395/qsoabsfind/blob/main/qsoabsfind/data/desi/desi_constants.py>`_ file; otherwise, the code will fail. If you want to use the default search parameters, you can run the tool without specifying the `constant-file` option.


A minimal example for the ``constants.py``
------------------------------------------

.. code-block:: python

    """
    data/desi/desi_constants.py

    This module defines search parameters. It includes general and survey-specific configurations
    for DESI quasar spectra.

    Used in:
    - data/desi/qso_test_spectra.py
    - To generate: data/desi/MgII_cat.fits and data/desi/CIV_cat.fits

    Usage:
        from qsoabsfind.config import load_constants
        search_parameters = load_constants(file_path)
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
        'logwave': False,  # Assume DESI-style linear scale by default
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
    - DESI spectra: `logwave=False` (linear wavelength grid); per-pixel resolution is computed.
    - MgII/CIV default parameters are optimized for DESI spectra
    Adjust as needed based on spectrum type.

    Fixed fractional error due to continuum normalization (5%)
    Used in estimating the uncertainty on column density
    arising from continuum placement uncertainties.
    Currently set empirically -- a more optimal estimate can be obtained
    by stacking continuum-normalized residual spectra in the observed frame
    and measuring the standard deviation from unity.
    """

.. note::

   The pipeline is not limited to the 9 built-in doublets. Any doublet system can be
   targeted by providing a constants file that defines the rest-frame wavelengths,
   oscillator strengths, and search window for the two transitions, following the
   structure shown above. The built-in systems (MgII, CIV, OVI, NV, SiIV, AlIII,
   FeII, CaII, NaI) benefit from thorough testing; custom systems are functional
   but have not been as extensively validated.

