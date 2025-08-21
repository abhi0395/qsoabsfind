"""
This script contains functions to read, append and write fits files.
"""
import os
from astropy.io import fits
import numpy as np
from astropy.table import Table

#Constants
from .constants import doublet_keys

def read_fits_file(fits_file, index=None):
    """
    Reads the FLUX, ERROR, WAVELENGTH, and METADATA extensions from the
    FITS file.

    Args:
        fits_file (str): Path to the FITS file containing QSO spectra.
        index (int, list, or np.ndarray, optional): Index or indices of the rows to load. Default is None.

    Returns:
        tuple: A tuple containing the flux, error, wavelength, and metadata data.
    """
    ## Read the metadata Table first:
    metadata = Table.read(fits_file, hdu="METADATA") ## this preserves the units
    with fits.open(fits_file, memmap=True) as hdul:
        header = hdul[0].header
        wavelength = hdul['WAVELENGTH'].data  # Assuming wavelength is common for all spectra
        if index is None:
            flux = hdul['FLUX'].data
            error = hdul['ERROR'].data
        else:
            metadata = metadata[index] ## get metadata only for the input index (or indices)
            if isinstance(index, int):
                flux = hdul['FLUX'].data[index].flatten() # flux, error should be 1D for a single spectrum
                error = hdul['ERROR'].data[index].flatten()
            else:
                flux = hdul['FLUX'].data[index]
                error = hdul['ERROR'].data[index]

    return header, flux, error, wavelength, metadata

def save_results_to_fits(results, input_file, output_file, headers, absorber):
    """
    Save the absorber results to a FITS file along with the metadata of QSOs.

    Args:
        results (dict): The results dictionary.
        intput_file (str): The path to the input spectra FITS file.
        output_file (str): The path to the output FITS file.
        headers (dict): The headers to include in the FITS file.
        absorber (str): The absorber type (MgII or CIV).

    Returns:
        A fits file containing detected absorber properties in 'ABSORBER' HDU and
        corresponding QSO metadata in 'METADATA' HDU.
    """
    if absorber not in doublet_keys:
        raise ValueError(f"Unsupported absorber, must be in {doublet_keys.keys()}")
    else:
        EW_TOTAL = f'{absorber.upper()}_EW_TOTAL'
        l1, l2 = doublet_keys[absorber][0].upper(), doublet_keys[absorber][1].upper()
        sn_1, sn_2 = f'SN_{l1}', f'SN_{l2}'
        EW_1, EW_2 = f'{l1}_EW', f'{l2}_EW'
        VDISP1, VDISP2 = f'{l1}_VDISP', f'{l2}_VDISP'

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
        fits.Column(name=sn_2, format='D', array=np.array(results['sn_2'])),
        fits.Column(name=VDISP1, format='D', unit='km s-1', array=np.array(results['vel_disp1'])),
        fits.Column(name=VDISP2, format='D', unit='km s-1', array=np.array(results['vel_disp2'])),
        fits.Column(name='DELTA_CHI2', format='D', array=np.array(results['delta_chi2'])),
    ], name='ABSORBER')

    hdr = fits.Header()
    for key, header in headers.items():
        hdr[key] = (header["value"], header["comment"])

    # load the QSO METADATA
    _,_, _, _, metadata = read_fits_file(input_file, index=np.array(results['index_spec']))
    qso_hdu = fits.BinTableHDU(metadata, name='METADATA')
    hdul = fits.HDUList([fits.PrimaryHDU(header=hdr), hdu, qso_hdu])

    hdul.writeto(output_file, overwrite=True)
    print(f'INFO: ouptut file {output_file} written.')

def append_table_to_fits(filename, table, hdu_name):
    """
    Append an Astropy Table as a new BinTableHDU to a FITS file.

    Args:
        filename (str): Path to the FITS file to write to.
        table (astropy.table.Table): Table to append.
        hdu_name (str): Name of the new HDU (used for identification).

    """
    if not isinstance(table, Table):
        raise TypeError("Provided 'table' must be an astropy.table.Table object")

    # Create the new HDU
    new_hdu = fits.BinTableHDU(data=table, name=hdu_name)

    if os.path.exists(filename):
        # Open existing file and append
        with fits.open(filename, mode='update') as hdul:
            hdul.append(new_hdu)
            hdul.flush()
    else:
        raise ValueError(f"ERROR: {filename} does not exist")


def read_any_fits_file(filename, hdu_name):
    """Read any fits file given filename and hdu_name.

    Args:
        filename (str): fits filepath
        hdu_name (str or int): HDU extension name or number
    Returns:
        astropy data: Table for BinTableHDU, data array for Image/Primary HDU
    """
    from astropy.io import fits
    from astropy.table import Table

    with fits.open(filename) as hdul:
        # Get the specific HDU
        hdu = hdul[hdu_name]
        hdr = hdul[0].header # Primary Headers

        # Check HDU type and read accordingly
        if isinstance(hdu, fits.BinTableHDU):
            return hdr, Table.read(filename, hdu=hdu_name)
        elif isinstance(hdu, fits.PrimaryHDU):
            return hdr, hdu.data
        elif isinstance(hdu, fits.ImageHDU):
            return hdr, hdu.data
        else:
            print(f"Reading {type(hdu).__name__} '{hdu_name}' as data array")
            return hdr, hdu.data


