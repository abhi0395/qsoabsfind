File formats
============

Input FITS File Structure
-------------------------

The input `fits file` must have the following HDU extensions:

- ``FLUX``: Should ideally contain the residual spectra (usually the flux/continuum, i.e., the continuum normalized spectra).
- ``WAVELENGTH``: Observed wavelength (in Angstroms).
- ``ERROR``: Error on residuals.
- ``METADATA``: Spectral details (such as Z_QSO, RA_QSO, DEC_QSO).

I have also provided two example QSO spectra FITS files. You can use these files to test an example run as described below.

    - ``data/sdss/qso_test_spectra.fits``, which contains 100 continuum-normalized SDSS QSO spectra.
    - ``data/desi/qso_test_spectra.fits``, which contains 100 continuum-normalized DESI DR1 QSO spectra.

Output FITS File Structure
--------------------------

The **output** FITS file always contains four HDUs. An optional fifth HDU is added when ``--coldens-dv`` is used,
and when ``--zabs-known-file`` is used the ``ABSORBER`` HDU gains an extra column:

**1) PRIMARY** HDU contains user-supplied key/value headers passed via ``--headers``.

**2) ABSORBER** HDU contains one row per detected absorber (or one row per validated redshift
when using ``--zabs-known-file``). Columns:

- ``INDEX_SPEC``: (*int*), Index of the QSO spectrum in the input file.
- ``Z_ABS``: (*float*), Redshift of the absorber. Sentinel values: ``-1`` means the search could not be attempted (too few pixels or doublet outside wavelength coverage); ``0`` means the search ran but nothing was found.
- ``${METAL}_${LINE}_EW``: (*float*), Rest-frame equivalent width of each doublet line (e.g. ``MGII_2796_EW``, ``MGII_2803_EW``) in Angstroms.
- ``${METAL}_${LINE}_EW_ERROR``: (*float*), Uncertainty on the rest-frame EW in Angstroms.
- ``${METAL}_EW_TOTAL``: (*float*), Sum of EWs of both doublet lines in Angstroms.
- ``${METAL}_EW_TOTAL_ERROR``: (*float*), Uncertainty on the total EW in Angstroms.
- ``Z_ABS_ERR``: (*float*), Uncertainty on the absorber redshift.
- ``GAUSS_FIT``: (*float array[6]*), Double-Gaussian rest-frame fit parameters [amp1, centre1, sigma1, amp2, centre2, sigma2].
- ``GAUSS_FIT_STD``: (*float array[6]*), Uncertainties on the Gaussian fit parameters.
- ``SN_${METAL}_${LINE}``: (*float*), Signal-to-noise ratio of each doublet line.
- ``${METAL}_${LINE}_VDISP``: (*float*), Instrumental-resolution-corrected rest-frame velocity dispersion of each line in km/s. Zero for unresolved lines.
- ``DOUBLET_RATIO``: (*float*), Measured equivalent-width ratio ``EW_stronger / EW_weaker`` for the doublet.
- ``DOUBLET_RATIO_THIN_LIMIT``: (*float*), Expected thin-limit ratio from atomic physics, ``(f*lambda^2)_stronger / (f*lambda^2)_weaker``.
- ``VDISP_DIFF``: (*float*), Absolute difference between the two fitted velocity dispersions, ``|VDISP_1 - VDISP_2|`` (km/s).
- ``VDISP_RATIO``: (*float*), Ratio of larger to smaller fitted velocity dispersion, ``max(VDISP_1,VDISP_2)/min(VDISP_1,VDISP_2)``.
- ``VDISP_FLAG``: (*int*), Velocity-dispersion consistency flag: ``1`` if ``VDISP_RATIO`` exceeds the configured threshold, else ``0``.
- ``DR_FLAG``: (*int*), Doublet-ratio nominality flag: ``1`` if ``DOUBLET_RATIO`` is outside the nominal physical range, else ``0``.
- ``QUALITY_FLAG``: (*int*), Combined diagnostic flag: ``0`` clean, ``1`` suspicious velocity-dispersion ratio only, ``2`` suspicious doublet ratio only, ``3`` both suspicious.
- ``DELTA_CHI2``: (*float*), Improvement in chi2 between the double-Gaussian model and a flat continuum (null hypothesis).
- ``ZABS_KNOWN``: (*float*, **only present when** ``--zabs-known-file`` **is used**), the input known redshift supplied for validation.

**3) METADATA** HDU contains all metadata columns from the input file's ``METADATA`` extension,
with one row per entry in the ``ABSORBER`` HDU.

**4) QSO_INFO** HDU summarises every spectrum that was processed, regardless of whether an
absorber was detected. Columns:

- ``INDEX_SPEC``: (*int*), Spectrum index in the input file.
- ``Z_QSO``: (*float*), QSO redshift from the input metadata.
- ``IS_QSO_AVAILABLE``: (*bool*), ``True`` if the search could be attempted (doublet wavelengths fall inside the spectrum's wavelength coverage and enough pixels were available); ``False`` when the spectrum was unsearchable.

**5) COLUMN_DENSITY** HDU is optional:

If the ``--coldens-dv`` option is provided when running ``qsoabsfind``, the code also calculates the total column density of each detected doublet using the apparent optical depth method.
following `Savage & Sembach (1991) <https://ui.adsabs.harvard.edu/abs/1991ApJ...379..245S/abstract>`_.

This optional HDU will contain:

- ``LOG10N``: (*float*), log of **total column density** (in cm\ :sup:`-2`), calculated from apparent optical depth method.
- ``SIG_LOG10N``: (*float*), uncertainty on log of **total column density** (in cm\ :sup:`-2`), calculated from apparent optical depth method.
- ``SATURATION``: (*int*), saturation flag, 1: saturated, 0: unsaturated
- ``fN``: (*int*), Column density measurement method, 1: WEIGHTED MEAN, 2: FIRST, 3: SECOND, 4: Corrected weak line (partial saturation), 5: Lower limit from weak line (strong saturation), 6: Lower limit from strong (strong saturation and weak is not available) -1: FAIL.
