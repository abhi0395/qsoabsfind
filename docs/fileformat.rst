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
- ``DELTA_CHI2``: (*float*), Improvement in chi2 between the double-Gaussian model and a flat continuum (null hypothesis).
- ``ZABS_KNOWN``: (*float*, **only present when** ``--zabs-known-file`` **is used**), the input known redshift supplied for validation.

**3) METADATA** HDU contains all metadata columns from the input file's ``METADATA`` extension,
with one row per entry in the ``ABSORBER`` HDU.

**4) QSO_INFO** HDU summarises every spectrum that was processed, regardless of whether an
absorber was detected. Columns:

- ``INDEX_SPEC``: (*int*), Spectrum index in the input file.
- ``Z_QSO``: (*float*), QSO redshift from the input metadata.
- ``IS_QSO_AVAILABLE``: (*bool*), ``True`` if the search could be attempted (doublet wavelengths fall inside the spectrum's wavelength coverage and enough pixels were available); ``False`` when the spectrum was unsearchable.
- ``SNR_QSO``: (*float*), Signal-to-noise ratio of the QSO spectrum in the absorber search window (using statistics passed by user, mean, median or weighted, HDU0 will have the information.).

**5) COLUMN_DENSITY** HDU is optional:

If the ``--coldens-dv`` option is provided when running ``qsoabsfind``, the code also calculates the total column density of each detected doublet using the apparent optical depth method.
following `Savage & Sembach (1991) <https://ui.adsabs.harvard.edu/abs/1991ApJ...379..245S/abstract>`_.

This optional HDU will contain:

- ``LOG10N``: (*float*), logarithm of the total column density,
  :math:`\log_{10}[N/(\mathrm{cm}^{-2})]`, measured using the apparent optical
  depth method.

- ``SIG_LOG10N``: (*float*), 1-\ :math:`\sigma` uncertainty on ``LOG10N``,
  in dex.

- ``SATURATION``: (*int*), saturation-status flag:

  - ``0``: no significant evidence for unresolved saturation.
  - ``1``: unresolved saturation detected; Savage & Sembach (1991)
    correction applied.
  - ``2``: strong/floor saturation, or saturation outside the calibrated
    correction range; reported column density is a lower limit.
  - ``3``: saturation cannot be determined because only one transition has
    a usable column-density measurement.
  - ``-2``: inconsistent doublet, where the weaker transition gives a
    significantly smaller apparent column density than the stronger
    transition.
  - ``-1``: column-density measurement failed.

- ``fN``: (*int*), column-density measurement method:

  - ``1``: inverse-variance weighted mean of both transitions.
  - ``2``: first transition only.
  - ``3``: second transition only.
  - ``4``: weaker transition corrected for unresolved saturation using
    Savage & Sembach (1991).
  - ``5``: lower limit from the weaker transition.
  - ``6``: lower limit from the stronger transition when the weaker
    transition is unavailable.
  - ``7``: inconsistent doublet.
  - ``-1``: column-density measurement failed.

- ``LOWER_LIMIT``: (*int*), lower-limit flag. ``1`` indicates that the
  reported column density is a lower limit, while ``0`` indicates a finite
  column-density measurement.

- ``DELTA_LOGN``: (*float*), difference between the apparent column
  densities measured from the weaker and stronger transitions,
  :math:`\log_{10}N_{\mathrm{weak}}-\log_{10}N_{\mathrm{strong}}`, in dex.

- ``SIG_DELTA_LOGN``: (*float*), statistical 1-\ :math:`\sigma` uncertainty
  on ``DELTA_LOGN``, in dex.

- ``NPIX_SAT_STRONG``: (*int*), number of pixels in the stronger transition
  whose normalized flux reaches or falls below the adopted AODM flux floor.

- ``NPIX_SAT_WEAK``: (*int*), number of pixels in the weaker transition
  whose normalized flux reaches or falls below the adopted AODM flux floor.