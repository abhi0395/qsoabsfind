qsoabsfind
==========

Instructions
-------------

Please read and perform the following steps.

Running Help Command
--------------------

To get an overview of the tool's functionality and options, run:

::

    qsoabsfind --help

Input FITS File Structure
-------------------------

The input `fits file` must have the following HDU extensions:

- **FLUX**: Should ideally contain the residual spectra (usually the flux/continuum, i.e., the continuum normalized spectra).
- **WAVELENGTH**: Observed wavelength (in Angstroms).
- **ERROR**: Error on residuals.
- **METADATA**: Spectral details (such as Z_QSO, RA_QSO, DEC_QSO).

Constant File (Optional)
------------------------

Before using your own constant file, please set an environment variable `QSO_CONSTANTS_FILE` in your `bashrc` or `zshrc` file and point it to the `qsoabsfind.constants` file. As the code loads the constants from new file dynamically, it is important to define this environment variable.

The user-defined **constant-file** must follow the same structure as the `qsoabsfind.constants` file, otherwise, the code will fail. If you want to use the default search parameters, you can run the tool without the `constant-file` option.

Example Usage
----------------

Run `qsoabsfind` with the required FITS file. If using a custom constant file, include it in the command:


    qsoabsfind \
        --input input_fits_file \
        --constant-file constant_file \
        --output output_fits_file \
        --absorber $absorber

To run the absorber search module, replace the placeholder paths as follows:

- ``input_fits_file``: Input QSO spectra FITS file
- ``constant_file``: Your constants file (e.g., ``data/sdss/sdss_constants.py`` or ``data/desi/desi_constants.py``) or your customized file
- ``output_fits_file``: Output filename resulting absorber catalog FITS file
- `absorber`: absorber (MgII or CIV)

Quick Example Runs
------------------

**For SDSS spectra** (e.g., MgII or CIV search):

    qsoabsfind \
        --input data/sdss/qso_test_spectra.fits \
        --constants data/sdss/sdss_constants.py \
        --absorber MgII \
        --output output_fits_file

**For DESI spectra** (e.g., MgII or CIV search):

    qsoabsfind \
        --input data/desi/qso_test_spectra.fits \
        --constants data/desi/desi_constants.py \
        --absorber MgII \
        --output output_fits_file

**Note**
---------

   Output catalogs for MgII and CIV absorber searches in both SDSS and DESI test spectra
   are already saved in the ``data/sdss/`` and ``data/desi/`` directories, respectively.


Output FITS File Structure
--------------------------

The **output** `fits file` will have two HDUs `ABSORBER` and `METADATA`:

**ABSORBER** HDU will contain following structured data:

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
- **${METAL}_${LINE}_VDISP**: Rest-frame instrumental-resolution-corrected velocity dispersion of each line (e.g., MgII 2796, 2803 or CIV 1548, 1550) in km/s. Can be **zero** for unresolved lines.

**METADATA** HDU will contain every metadata (corresponding to each absorber) that is available in input spectra file.


| Thanks,
| Abhijeet Anand
| Lawrence Berkeley National Lab
