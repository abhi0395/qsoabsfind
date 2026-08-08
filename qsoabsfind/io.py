"""
This script contains functions to read, append and write fits files.
"""
import os
import logging
from astropy.io import fits
import numpy as np
from astropy.table import Table

logger = logging.getLogger(__name__)

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

def _build_qso_info_hdu(input_file, spec_indices, results):
    spec_arr = np.array(list(spec_indices))
    metadata = Table.read(input_file, hdu="METADATA")
    metadata = Table(metadata[spec_arr])
    if 'Z' in metadata.colnames:
        metadata.rename_column('Z', 'Z_QSO')
    z_qso = np.array(metadata['Z_QSO'])
    # IS_QSO_AVAILABLE is True when the search could run (z_abs != -1),
    # regardless of whether an absorber was actually found. It is False
    # only when the spectrum had too few pixels or the doublet fell outside
    # the wavelength coverage (default value -1) or snr cut was not satisfied in case snr_cut was passed in kwargs argument.
    unsearchable = set(results.get('unsearchable_indices', []))
    is_available = np.array([int(idx) not in unsearchable for idx in spec_arr], dtype=bool)
    snr_qso_map = results.get('snr_qso_map', {})
    snr_qso = np.array([snr_qso_map.get(int(idx), -1.0) for idx in spec_arr], dtype=np.float32)
    return fits.BinTableHDU.from_columns([
        fits.Column(name='INDEX_SPEC', format='K', array=spec_arr),
        fits.Column(name='Z_QSO', format='D', array=z_qso),
        fits.Column(name='IS_QSO_AVAILABLE', format='L', array=is_available),
        fits.Column(name='SNR_QSO', format='E', array=snr_qso),
    ], name='QSO_INFO')


def save_results_to_fits(results, input_file, output_file, headers, absorber, spec_indices=None):
    """
    Save the absorber results to a FITS file along with the metadata of QSOs.

    Args:
        results (dict): The results dictionary from :func:`parallel_convolution_search`.
        input_file (str): The path to the input spectra FITS file.
        output_file (str): The path to the output FITS file.
        headers (dict): The headers to include in the FITS file.
        absorber (str): The absorber type (e.g. MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, NaI).
        spec_indices (list or array, optional): All spectrum indices that were processed.
            When provided, a ``QSO_INFO`` BinTableHDU with columns ``INDEX_SPEC``,
            ``Z_QSO``, and ``IS_QSO_AVAILABLE`` is appended. ``IS_QSO_AVAILABLE`` is
            ``True`` whenever the search could run (even if no absorber was found) and
            ``False`` when the spectrum was unsearchable (too few pixels or doublet outside
            wavelength coverage). Also triggers the ``ZABS_KNOWN`` column in the
            ``ABSORBER`` HDU when the results dict contains that key. Default is None.

    Returns:
        None: Writes a FITS file with an ``ABSORBER`` BinTableHDU containing
        detected absorber properties and a ``METADATA`` BinTableHDU with
        corresponding QSO metadata.
    """
    if absorber not in doublet_keys:
        raise ValueError(
            f"Absorber '{absorber}' not found in doublet_keys. "
            f"Built-in absorbers: {list(doublet_keys.keys())}. "
            "To use a custom doublet, add it to your constants file "
            "(see docs/paramfile.rst for the required format)."
        )
    else:
        EW_TOTAL = f'{absorber.upper()}_EW_TOTAL'
        l1, l2 = doublet_keys[absorber][0].upper(), doublet_keys[absorber][1].upper()
        sn_1, sn_2 = f'SN_{l1}', f'SN_{l2}'
        EW_1, EW_2 = f'{l1}_EW', f'{l2}_EW'
        VDISP1, VDISP2 = f'{l1}_VDISP', f'{l2}_VDISP'
        VDISP1_ERR, VDISP2_ERR = f'{l1}_VDISP_ERR', f'{l2}_VDISP_ERR'
        DCHI2_1, DCHI2_2 = f'DELTA_CHI2_{l1}', f'DELTA_CHI2_{l2}'
        REDCHI2 = f'REDCHI2_FIT'

    absorber_cols = [
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
        fits.Column(name=VDISP1_ERR, format='D', unit='km s-1', array=np.array(results['vel_disp1_err'])),
        fits.Column(name=VDISP2_ERR, format='D', unit='km s-1', array=np.array(results['vel_disp2_err'])),
        fits.Column(name=DCHI2_1, format='D', array=np.array(results['delta_chi2_line1'])),
        fits.Column(name=DCHI2_2, format='D', array=np.array(results['delta_chi2_line2'])),
        fits.Column(name=REDCHI2, format='D', array=np.array(results['pure_redchi2'])),
    ]
    if 'zabs_known' in results:
        absorber_cols.append(
            fits.Column(name='ZABS_KNOWN', format='D', array=np.array(results['zabs_known']))
        )
    hdu = fits.BinTableHDU.from_columns(absorber_cols, name='ABSORBER')

    hdr = fits.Header()
    for key, header in headers.items():
        hdr[key] = (header["value"], header["comment"])

    # Primay header
    primary_hdu = fits.PrimaryHDU(header=hdr)
    primary_hdu.header['EXTNAME'] = 'PRIMARY'

    # load the QSO METADATA
    _,_, _, _, metadata = read_fits_file(input_file, index=np.array(results['index_spec']))
    qso_hdu = fits.BinTableHDU(metadata, name='METADATA')

    hdu_list = [primary_hdu, hdu, qso_hdu]
    if spec_indices is not None:
        hdu_list.append(_build_qso_info_hdu(input_file, spec_indices, results))
    fits.HDUList(hdu_list).writeto(output_file, overwrite=True)
    logger.info("output file %s written.", output_file)

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


