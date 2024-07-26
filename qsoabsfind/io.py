import fitsio
import numpy as np

def read_fits_file(fits_file, index=None):
    """
    Reads the FLUX, ERROR, WAVELENGTH, and TGTDETAILS extensions from the FITS file.
    If index is provided, only loads the required rows.

    Parameters:
    fits_file (str): Path to the FITS file containing QSO spectra.
    index (int, list, or np.ndarray, optional): Index or indices of the rows to load. Default is None.

    Returns:
    tuple: A tuple containing the flux, error, wavelength, and tgtdetails data.
    """
    with fitsio.FITS(fits_file) as fits:
        if index is None:
            flux = fits['FLUX'].read()
            error = fits['ERROR'].read()
            wavelength = fits['WAVELENGTH'].read()
            tgtdetails = fits['TGTDETAILS'].read()
        else:
            if isinstance(index, int):
                index = [index]
            elif isinstance(index, (list,np.ndarray)):
                index = index.tolist()
            flux = fits['FLUX'].read(rows=index)
            error = fits['ERROR'].read(rows=index)
            wavelength = fits['WAVELENGTH'].read()  # Assuming wavelength is the same for all spectra
            tgtdetails = fits['TGTDETAILS'].read(rows=index)
    return flux, error, wavelength, tgtdetails

def save_results_to_fits(results, output_file, headers, absorber):
    """
    Save the results to a FITS file.

    Parameters:
    results (dict): The results dictionary.
    output_file (str): The path to the output FITS file.
    headers (dict): The headers to include in the FITS file.
    absorber (str): The absorber type (MgII or CIV).
    """
    EW_TOTAL = f'{absorber}_EW_TOTAL'
    if absorber == 'MgII':
        sn_1 = 'SN_MGII_2796'
        sn_2 = 'SN_MGII_2803'
        EW_1 = 'MGII_2796_EW'
        EW_2 = 'MGII_2803_EW'
    elif absorber == 'CIV':
        sn_1 = 'SN_CIV_1548'
        sn_2 = 'SN_CIV_1550'
        EW_1 = 'CIV_1548_EW'
        EW_2 = 'CIV_1550_EW'
    else:
        raise ValueError(f"Unsupported absorber: {absorber}")

    hdu = fits.BinTableHDU.from_columns([
        fits.Column(name='INDEX_SPEC', format='K', array=np.array(results['index_spec'])),
        fits.Column(name='Z_ABS', format='D', array=np.array(results['z_abs'])),
        fits.Column(name='GAUSS_FIT', format='6D', array=np.array(results['gauss_fit'])),
        fits.Column(name='GAUSS_FIT_STD', format='6D', array=np.array(results['gauss_fit_std'])),
        fits.Column(name=f'{EW_1}', format='D', array=np.array(results['ew_1_mean'])),
        fits.Column(name=f'{EW_2}', format='D', array=np.array(results['ew_2_mean'])),
        fits.Column(name=ew_total, format='D', array=np.array(results['ew_total_mean'])),
        fits.Column(name=f'{EW_1}_ERROR', format='D', array=np.array(results['ew_1_error'])),
        fits.Column(name=f'{EW_2}_ERROR', format='D', array=np.array(results['ew_2_error'])),
        fits.Column(name=f'{EW_TOTAL}_ERROR', format='D', array=np.array(results['ew_total_error'])),
        fits.Column(name='Z_ABS_ERR', format='D', array=np.array(results['z_abs_err'])),
        fits.Column(name=sn_1, format='D', array=np.array(results['sn_1'])),
        fits.Column(name=sn_2, format='D', array=np.array(results['sn_2']))
    ], name='ABSORBER')

    hdr = fits.Header()
    for key, value in headers.items():
        hdr[key.upper()] = value

    hdul = fits.HDUList([fits.PrimaryHDU(header=hdr), hdu])
    hdul.writeto(output_file, overwrite=True)
