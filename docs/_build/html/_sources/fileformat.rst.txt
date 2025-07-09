File formats
============

Input FITS File Structure
-------------------------

The input `fits file` must have the following HDU extensions:

- **FLUX**: Should ideally contain the residual spectra (usually the flux/continuum, i.e., the continuum normalized spectra).
- **WAVELENGTH**: Observed wavelength (in Angstroms).
- **ERROR**: Error on residuals.
- **METADATA**: Spectral details (such as Z_QSO, RA_QSO, DEC_QSO).

I have also provided two example QSO spectra FITS files. 1) `data/sdss/qso_test_spectra.fits`, which contains 500 continuum-normalized SDSS QSO spectra. 2) `data/desi/qso_test_spectra.fits`, which contains 500 continuum-normalized DESI DR1 QSO spectra. You can use these files to test an example run as described below.

Constant File (Optional)
------------------------

Before using your own constants file, please set an environment variable `QSO_CONSTANTS_FILE` in your `bashrc` or `zshrc` file, and point it to the `qsoabsfind.constants` file. Since the code dynamically loads constants from a new file, it is important to define this environment variable.

The user-defined **constants file** must follow the same structure as the `qsoabsfind.constants <https://github.com/abhi0395/qsoabsfind/blob/main/qsoabsfind/constants.py>`_ file; otherwise, the code will fail. If you want to use the default search parameters, you can run the tool without specifying the `constant-file` option.

Output FITS File Structure
--------------------------

The **output** `fits file` will have two (or three, optional) HDUs **ABSORBER** and **METADATA** or, (**COLUMN_DENSITY**, optional):

**1) ABSORBER** HDU will contain the following structured data:

- **INDEX_SPEC**: (int), Index of quasar (can be used to read the RA, DEC, and Z of QSOs).
- **Z_ABS**: (float), Redshift of absorber.
- **${METAL}_${LINE}_EW**: (float), Rest-frame equivalent widths (EWs) of absorber lines (e.g., MgII 2796, 2803 or CIV 1548, 1550) in Angstroms.
- **${METAL}_${LINE}_EW_ERROR**: (float), Uncertainties in rest-frame EWs of absorber lines in Angstroms.
- **Z_ABS_ERR**: (float), Measured error in the redshift of the absorber.
- **GAUSS_FIT**: (float array), Rest-frame fitting parameters of a double Gaussian to the absorber doublet (the width can be used to measure the velocity dispersion).
- **GAUSS_FIT_STD**: (float array), Uncertainties in rest-frame fitting parameters of the double Gaussian to the absorber doublet.
- **SN_${METAL}_${LINE}**: (float), Signal-to-noise ratio of the lines.
- **${METAL}_EW_TOTAL**: (float), Total EW of the lines in Angstroms.
- **${METAL}_EW_TOTAL_ERROR**: (float), Uncertainties in total EW of the lines in Angstroms.
- **${METAL}_${LINE}_VDISP**: (float), Rest-frame instrumental-resolution-corrected velocity dispersion of each line (e.g., MgII 2796, 2803 or CIV 1548, 1550) in km/s. Can be **zero** for unresolved lines.


**2) METADATA** HDU will contain all the metadata (corresponding to each absorber) available in the input spectra file.

**3) COLUMN_DENSITY** HDU is optional:

If the ``--coldens`` option is provided when running ``qsoabsfind``, the code also calculates the total column density of each detected doublet using the apparent optical depth method.
This implementation follows the methodology described by `Savage & Sembach (1991) <https://ui.adsabs.harvard.edu/abs/1991ApJ...379..245S/abstract>`_.

This optional HDU will contain:

- **LOG10N**: (float), log of **total column density** (in cm<sup>-2</sup>), calculated from apparent optical depth method.
- **SIG_LOG10N**: (float), uncertainty on log of **total column density** (in cm<sup>-2</sup>), calculated from apparent optical depth method.
- **SATURATION**: (int), saturation flag, 1: saturated, 0: unsaturated
- **fN**: (int), Column density measurement method, 1: WEIGHTED MEAN, 2: FIRST, 3: SECOND, 4: Corrected weak line (partial saturation), 5: Lower limit from weak line (strong saturation), 6: Lower limit from strong (strong saturation and weak is not available) -1: FAIL.
