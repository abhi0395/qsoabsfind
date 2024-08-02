File formats
============

Input FITS File Structure
-------------------------

The input `fits file` must have the following HDU extensions:

- **FLUX**: Should ideally contain the residual spectra (usually the flux/continuum, i.e., the continuum normalized spectra).
- **WAVELENGTH**: Observed wavelength (in Angstroms).
- **ERROR**: Error on residuals.
- **METADATA**: Spectral details (such as Z_QSO, RA_QSO, DEC_QSO).

I have also provided an example QSO spectra FITS file, `data/qso_test.fits`, which contains 500 continuum-normalized SDSS QSO spectra. You can use this file to test an example run as described below.

Constant File (Optional)
------------------------

Before using your own constants file, please set an environment variable `QSO_CONSTANTS_FILE` in your `bashrc` or `zshrc` file, and point it to the `qsoabsfind.constants` file. Since the code dynamically loads constants from a new file, it is important to define this environment variable.

The user-defined **constants file** must follow the same structure as the `qsoabsfind.constants <https://github.com/abhi0395/qsoabsfind/blob/main/qsoabsfind/constants.py>`_ file; otherwise, the code will fail. If you want to use the default search parameters, you can run the tool without specifying the `constant-file` option.

Then run `qsoabsfind` with the required FITS file. If using a custom constant file, include it in the command:

::

    qsoabsfind --input-fits-file data/qso_test.fits \
               --n-qso 500 \
               --absorber MgII \
               --output test_MgII.fits \
               --headers SURVEY=SDSS AUTHOR=YOUR_NAME \
               --n-tasks 16 \
               --ncpus 4
               --constant-file path_to_your_file

Output FITS File Structure
--------------------------

The **output** `fits file` will have the `ABSORBER` HDU, containing arrays such as:

- **INDEX_SPEC**: Index of quasar (can be used to read the RA, DEC, and Z of QSOs).
- **Z_ABS**: Redshift of absorber.
- **${METAL}_${LINE}_EW**: Rest-frame equivalent widths (EWs) of absorber lines (e.g., MgII 2796, 2803 or CIV 1548, 1550) in Angstroms.
- **${METAL}_${LINE}_EW_ERROR**: Uncertainties in rest-frame EWs of absorber lines in Angstroms.
- **Z_ABS_ERR**: Measured error in the redshift of the absorber.
- **GAUSS_FIT**: Rest-frame fitting parameters of double Gaussian to the absorber doublet (the width can be used to measure the velocity dispersion).
- **GAUSS_FIT_STD**: Uncertainties in rest-frame fitting parameters of double Gaussian to the absorber doublet.
- **SN_${METAL}_${LINE}**: Signal-to-noise ratio of the lines.
- **${METAL}_EW_TOTAL**: Total EW of the lines in Angstroms.
- **${METAL}_EW_TOTAL_ERROR**: Uncertainties in total EW of the lines in Angstroms.
