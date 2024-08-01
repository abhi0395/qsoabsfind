"""
This script contains a functions to read and write files.
"""

import astropy.io.fits as fits
import numpy as np

def read_fits_file(fits_file, index=None):
    """
    Reads the FLUX, ERROR, WAVELENGTH, and TGTDETAILS extensions from the
    FITS file.

    Args:
        fits_file (str): Path to the FITS file containing QSO spectra.
        index (int, list, or np.ndarray, optional): Index or indices of the rows to load. Default is None.

    Returns:
        tuple: A tuple containing the flux, error, wavelength, and tgtdetails data.
    """
    with fits.open(fits_file, memmap=True) as hdul:
        if index is None:
            flux = hdul['FLUX'].data
            error = hdul['ERROR'].data
            wavelength = hdul['WAVELENGTH'].data
            tgtdetails = hdul['TGTDETAILS'].data
        else:
            if isinstance(index, int):
                flux = hdul['FLUX'].data[index].flatten()
                error = hdul['ERROR'].data[index].flatten()
                tgtdetails = {name: hdul['TGTDETAILS'].data[index][name] for name in hdul['TGTDETAILS'].data.names}
            else:
                flux = hdul['FLUX'].data[index]
                error = hdul['ERROR'].data[index]
                tgtdetails = hdul['TGTDETAILS'].data[index]
            wavelength = hdul['WAVELENGTH'].data  # Assuming wavelength is common for all spectra
    return flux, error, wavelength, tgtdetails

def save_results_to_fits(results, output_file, headers, absorber):
    """
    Save the results to a FITS file.

    Args:
        results (dict): The results dictionary.
        output_file (str): The path to the output FITS file.
        headers (dict): The headers to include in the FITS file.
        absorber (str): The absorber type (MgII or CIV).
    """
    EW_TOTAL = f'{absorber.upper()}_EW_TOTAL'
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
        fits.Column(name=f'{EW_1}', format='D', unit='Angstrom', array=np.array(results['ew_1_mean'])),
        fits.Column(name=f'{EW_2}', format='D', unit='Angstrom', array=np.array(results['ew_2_mean'])),
        fits.Column(name=f'{EW_TOTAL}', format='D', unit='Angstrom', array=np.array(results['ew_total_mean'])),
        fits.Column(name=f'{EW_1}_ERROR', format='D', unit='Angstrom', array=np.array(results['ew_1_error'])),
        fits.Column(name=f'{EW_2}_ERROR', format='D', unit='Angstrom', array=np.array(results['ew_2_error'])),
        fits.Column(name=f'{EW_TOTAL}_ERROR', format='D', unit='Angstrom', array=np.array(results['ew_total_error'])),
        fits.Column(name='Z_ABS_ERR', format='D', array=np.array(results['z_abs_err'])),
        fits.Column(name=sn_1, format='D', array=np.array(results['sn_1'])),
        fits.Column(name=sn_2, format='D', array=np.array(results['sn_2']))
    ], name='ABSORBER')

    hdr = fits.Header()
    for key, header in headers.items():
        hdr[key] = (header["value"], header["comment"])

    hdul = fits.HDUList([fits.PrimaryHDU(header=hdr), hdu])
    hdul.writeto(output_file, overwrite=True)
