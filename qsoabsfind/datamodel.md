qsoabsfind
============

Instructions
-------------

Please read/perform following details.

`qsoabsfind --help`

Please go through all the functions and their instructions in case you have any doubt.

The **input** `fits file` must have a specific structure with following hdu extensions.

- `FLUX` Should contain ideally the residual spectra (usually the flux/continuum, i.e. the continuum normalized spectra)
- `WAVELENGTH` observed wavelength (in Ang)
- `ERROR` error on residuals
- `TGTDETALS` spectral details (like Z_QSO, RA_QSO, DEC_QSO...)

The user-defined **constant-file** must follow the same structure as the `qsoabsfind.constants`, otherwise the code will fail. In case, users want to use the default search parameters, they can tun without `constant-file` option.

The **output** `fits file` will have the hdu `ABSORBER`, that will arrays such as

- `INDEX_SPEC` index of quasar (can be used to read the RA, DEC ans Z of QSOs)
- `Z_ABS`: redshift of absorber
- `${METAL}_${line}_EW`: Rest-frame EWs of absorber lines (like MgII 2796, 2803 or CIV 1548, 1550) (in Ang)
- `${METAL}_${line}_EW_ERROR`: uncertainities in rest-frame EWs of absorber lines (like MgII 2796, 2803 or CIV 1548, 1550) (in Ang)
- `Z_ABS_ERR`: measured error in redshift of absorber
- `GAUSS_FIT`: rest-frame fitting parameters of double gaussian to the absorber doublet (the width can be used to measure the velocity dispersion)
- `GAUSS_FIT_STD`: uncertainities in  rest-frame fitting parameters of double gaussian to the absorber doublet (the width can be used to measure the velocity dispersion)
- `SN_${METAL}_${line}`: S/N of the lines
- `${metal}_EW_TOTAL`: Total EW of the lines (in Ang)
- `${metal}_EW_TOTAL_ERROR`: uncertainities in total EW of the lines (in Ang)


Thanks,  
Abhijeet Anand  
Lawrence Berkeley National Lab  

If you have any questions/suggestions, please feel free to write to abhijeetanand2011@gmail.com
