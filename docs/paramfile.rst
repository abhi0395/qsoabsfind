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
    Fixed fractional error due to continuum normalization (5%)
    Used in estimating the uncertainty on column density
    arising from continuum placement uncertainties.
    Currently set empirically -- a more optimal estimate can be obtained
    by stacking continuum-normalized residual spectra in the observed frame and measuring the standard deviation from unity.
    """

.. note::

   The pipeline is not limited to the 9 built-in doublets. Any doublet system can be
   targeted by providing a constants file that defines the rest-frame wavelengths,
   oscillator strengths, and search window for the two transitions, following the
   structure shown above. The built-in systems (MgII, CIV, OVI, NV, SiIV, AlIII,
   FeII, CaII, NaI) benefit from thorough testing; custom systems are functional
   but have not been as extensively validated.


Adding a Custom Absorber
------------------------

To search for a doublet not in the default list, add the following entries to your
constants file. The pipeline patches its internal registries at startup, so no
source-code changes are needed.

.. code-block:: python

    from qsoabsfind.constants import doublet_keys, lines, oscillator_parameters, amplitude_dict

    # --- Register the new doublet ---
    # Keys must be unique strings; the convention is <Name>_<wavelength>.
    doublet_keys['MyAbs'] = ('MyAbs_1234', 'MyAbs_5678')

    # Rest-frame wavelengths (Ang)
    lines['MyAbs_1234'] = 1234.56
    lines['MyAbs_5678'] = 5678.90

    # Oscillator strengths (f-values); required for column-density calculation
    oscillator_parameters['MyAbs_f1'] = 0.50   # stronger line
    oscillator_parameters['MyAbs_f2'] = 0.25   # weaker line

    # Convolution kernel amplitude ratio (optional; defaults to 0.5 if omitted)
    amplitude_dict['MyAbs'] = 0.5

Then pass ``--absorber MyAbs --constant-file my_constants.py`` on the command line.

.. warning::

   The search-window logic (``start_rest_wave`` / ``end_rest_wave``) and all
   selection thresholds in ``search_parameters`` are taken directly from your
   constants file, so make sure they are physically appropriate for the new doublet.
   The defaults were tuned for the built-in systems and may not be optimal.


YAML Configuration File
-----------------------

Instead of passing all arguments on the command line you can store them in a
YAML file and pass it with ``--config``. CLI flags always take precedence.

A fully annotated template (``data/example_config.yaml``):

.. code-block:: yaml

    # qsoabsfind configuration file
    # Pass this file with:  qsoabsfind --config /path/to/config.yaml
    #
    # Keys must match the CLI argument names with underscores (not dashes),
    # e.g. "input_fits_file" for --input-fits-file.
    #
    # Any argument supplied on the command line overrides the value here.

    # -- Required ------------------------------------------------------------------
    input_fits_file: /path/to/your/input/fits/file.fits
    absorber: MgII
    constant_file: /path/to/your/constants/file.py
    output: /path/to/your/output/file.fits

    # -- Optional ------------------------------------------------------------------
    n_qso: null        # null = run all; or e.g. "100", "1-1000", "1-1000:10"
    ncpus: 4
    verbose: false

    # Column density calculation: provide a float (km/s) to enable, or null to disable
    coldens_dv: 300   # e.g. +/-300.0 km/s window

    # Extra FITS headers in NAME=VALUE format (leave as [] or just comment these out if none needed)
    headers:
      - SURVEY=DESI
      - DR=DR1

    trapz_ew_sigma: 3.0   # n_sigma for trapezoidal EW measurement; set to null to disable

.. note::

    YAML keys use underscores, not dashes (e.g. ``input_fits_file`` for
    ``--input-fits-file``). Set optional keys to ``null`` to let argparse
    use its built-in default.

