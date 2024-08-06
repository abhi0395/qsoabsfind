"""
This script contains some utility functions.
"""

import time
#import logging
import numpy as np
from .config import load_constants
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.table import Table

# Configure logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the logging level for Matplotlib to WARNING to suppress DEBUG messages
#logging.getLogger('matplotlib').setLevel(logging.WARNING)

constants = load_constants()
lines, amplitude_dict, speed_of_light = constants.lines, constants.amplitude_dict, constants.speed_of_light

def elapsed(start, msg):
    """
    Prints the elapsed time since `start`.

    Args:
        start (float): The start time.
        msg (str): The message to print with the elapsed time.

    Returns:
        float: The current time.
    """
    end = time.time()
    if start is not None:
        print(f"{msg} {end - start:.2f} seconds")
    return end

def gauss_two_lines_kernel(x, a):
    """
    Defines the kernel function using double gaussian only.

    Args:
        x (numpy.ndarray): Kernel lambda array (user defined),
        a (numpy.ndarray): Kernel parameters, 6 parameters (amp, mean, and sigma for two Gaussian),

    Returns:
        numpy.ndarray: The kernel function (array of numbers).
    """
    a1 = a[0]
    a2 = a[3]

    norm_constant = -1#/((a1+a2)*(2*np.pi*a[1]**2)**0.5)

    return norm_constant * (-a1 * np.exp(-((x - a[1]) / a[2]) ** 2 / 2) - a2 * np.exp(-((x - a[4]) / a[5]) ** 2 / 2)) * 0.5

def convolution_fun(absorber, residual_arr_after_mask, width, amp_ratio=0.5, log=True, index=None):
    """
    Convolves the spectrum with a Gaussian kernel.

    Args:
        absorber (str): Type of absorber (e.g., 'MgII', 'CIV').
        residual_arr_after_mask (numpy.ndarray): Final residual array after masking.
        width (float): The width of the Gaussian kernel (decide base dupon width of real absorption feature).
        amp_ratio (float): Amplitude ratio for the Gaussian lines (default 0.5).
        log (bool): if log bins should be used for wavelength (dlam = 0.0001, default True)
        index (int): QSO index

    Returns:
        numpy.ndarray: The convolved residual array.
    """
    if absorber not in amplitude_dict:
        raise ValueError(f"Unsupported absorber type. Available types are: {list(amplitude_dict.keys())}")

    A_main = amplitude_dict[absorber]
    A_secondary = A_main * amp_ratio
    ct = 5
    if absorber == 'MgII':
        ker_parm = np.array([A_main, lines['MgII_2796'], width, A_secondary, lines['MgII_2803'], width])
        lam_ker_start = lines['MgII_2796']-ct*width # +/- 5sigma , #rest-frame
        lam_ker_end = lines['MgII_2803']+ct*width
    elif absorber == 'CIV':
        ker_parm = np.array([A_main, lines['CIV_1548'], width, A_secondary, lines['CIV_1550'], width])
        lam_ker_start = lines['CIV_1548']-ct*width # +/- 5sigma , #rest-frame
        lam_ker_end = lines['CIV_1550']+ct*width #
    else:
        raise ValueError(f"Unsupported absorber type for specific Args: {absorber}")
    if log:
        lam_ker = np.arange(np.log10(lam_ker_start), np.log10(lam_ker_end), 0.0001) #SDSS-like wavelength resolution
        lam_ker = 10**lam_ker
    else:
        lam_ker = np.arange(lam_ker_start, lam_ker_end, 0.8) # DESI-like wavelength resolution

    if len(lam_ker)>len(residual_arr_after_mask):
        lam_ker = lam_ker[0: len(residual_arr_after_mask)]

    gauss_kernel = gauss_two_lines_kernel(lam_ker, a=ker_parm)

    result = np.convolve(gauss_kernel, residual_arr_after_mask, mode='same')

    #check if input and output array size are same
    bad_conv = validate_sizes(result, residual_arr_after_mask, index)
    if bad_conv == 1:
        print(f"ERROR: Size mismatch detected in spec_index {index}")
    return result

def double_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    """
    Generates a double Gaussian function to fit absorption features in a
    given spectrum.

    Args:
        x (numpy.ndarray): List of wavelength points where the user wants to fit the model.
        amp1: Amplitude of the first Gaussian.
        mean1: Mean (center) of the first Gaussian.
        sigma1: Standard deviation (width) of the first Gaussian.
        amp2: Amplitude of the second Gaussian.
        mean2: Mean (center) of the second Gaussian.
        sigma2: Standard deviation (width) of the second Gaussian.

    Returns:
        numpy.ndarray: The function that fits the absorption feature using curve_fit.
    """
    return -amp1 * np.exp(-(x - mean1) ** 2 / (2 * sigma1 ** 2)) - amp2 * np.exp(-(x - mean2) ** 2 / (2 * sigma2 ** 2)) + 1

def single_gaussian(x, params):
    """
    Defines the fitting function to fit a single absorption line with a
    gaussian profile.

    Args:
        x (numpy.ndarray): Wavelength points where the user wants to fit the model.
        params (list or numpy.ndarray): Array of parameters [amp, mean, sigma].

    Returns:
        numpy.ndarray: The fitting function values.
    """
    amp, mean, sigma = params
    return -amp * np.exp(-((x - mean) / sigma) ** 2 / 2) + 1

def save_plot(x, y, plot_filename='qsoabsfind_plot.png', xlabel='X-axis', ylabel='Y-axis', title='Plot Title'):
    """
    Saves a plot of x vs y in the current working directory. If y is a list
    of arrays, each will be plotted.

    Args:
        x (array-like): The x data.
        y (array-like or list of array-like): The y data or list of y data arrays.
        plot_filename (str): The filename for the saved plot. Default is 'qsoabsfind_plot.png'.
        xlabel (str): The label for the x-axis. Default is 'X-axis'.
        ylabel (str): The label for the y-axis. Default is 'Y-axis'.
        title (str): The title of the plot. Default is 'Plot Title'.
    """
    # Create the plot
    plt.figure()

    if isinstance(y, list):
        for y_data in y:
            plt.plot(x, y_data, ls='-', lw=1.5)
    else:
        plt.plot(x, y, ls='-', lw=1.5)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)

    # Get the current working directory
    current_dir = os.getcwd()

    # Define the full path for the plot
    plot_path = os.path.join(current_dir, plot_filename)

    # Save the plot
    plt.savefig(plot_path)
    plt.close()

    print(f"Plot saved as {plot_path}")

def combine_fits_files(directory, output_filename='combined.fits'):
    """
    Combine HDUs from all FITS files in a directory into a single FITS file, preserving headers
    and saves the final combined file in the same directory.

    Args:
        directory: str, path to the directory containing FITS files (absorber file for each spectra file).
        output_filename: str, name of the output combined FITS file (default: 'combined.fits').

    Note:
        This function assumes that all the corresponding absorber fits files are in the input directory,
        and follow the same structure as the output of qsoabsfind. Also there should not be any other
        FITS file there. Otherwise, script will fail.
    """
    # Ensure the directory exists
    if not os.path.isdir(directory):
        raise FileNotFoundError(f"The directory {directory} does not exist.")

    # List to hold all HDUs for the combined FITS file
    combined_hdul = fits.HDUList()

    # Track added extension names to avoid duplicates
    added_extnames = set()

    # Iterate over all FITS files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".fits"):
            file_path = os.path.join(directory, filename)
            print(f"Processing file: {file_path}")

            # Open the FITS file
            with fits.open(file_path, memmap=True) as hdul:
                # Iterate through each HDU in the file
                for hdu in hdul:
                    # If it's the Primary HDU or a unique extension, add it to the combined list
                    if isinstance(hdu, fits.PrimaryHDU) or hdu.name not in added_extnames:
                        # Copy the HDU to preserve data and header
                        combined_hdul.append(hdu.copy())
                        added_extnames.add(hdu.name)
                        print(f"Added HDU '{hdu.name}' from {filename}")

    # Construct the output file path
    output_file_path = os.path.join(directory, output_filename)

    # Save the combined HDU list to a new FITS file
    combined_hdul.writeto(output_file_path, overwrite=True)
    print(f"Combined FITS file saved as {output_file_path}")

def validate_sizes(conv_arr, unmsk_residual, spec_index):
    """
    Validate that all arrays have the same size.

    Args:
        conv_arr (np.ndarray): Convolved array.
        unmsk_residual (np.ndarray): Unmasked residual array.
        spec_index (int): QSO index

    Returns:
        assertion errors
    """
    bad_conv=0
    try:
        assert conv_arr.size == unmsk_residual.size
    except AssertionError:
        bad_conv=1
        # logging.error(f"Size mismatch detected in spec_index {spec_index}")
        # logging.debug(f"conv_arr size: {conv_arr.size}, unmsk_residual size: {unmsk_residual.size}")
        # raise
    return bad_conv

def vel_dispersion(c1, c2, sigma1, sigma2, resolution):
    """
    Calculates velocity dispersion using Gaussian quadrature.

    Args:
        c1 (float): fitted line center 1 (in Ang).
        c2 (float): fitted line center 2 (in Ang).
        sigma1 (float): fitted width 1 (in Ang).
        sigma2 (float): fitted width 2 (in Ang).
        resoultion (float): instrumental resolution (in km/s).

    Returns:
        resolution corrected velocity dispersion
    """

    v1_sig = sigma1 / c1 * speed_of_light
    v2_sig = sigma2 / c2 * speed_of_light

    del_v1_sq = v1_sig**2 - resolution**2
    del_v2_sq = v2_sig**2 - resolution**2

    # Correct for instrumental resolution
    if del_v1_sq > 0 and del_v2_sq > 0:
        corr_del_v1_sq = np.sqrt(del_v1_sq)
        corr_del_v2_sq = np.sqrt(del_v2_sq)
    else:
        corr_del_v1_sq  = 0.0  # Set to 0 if the observed width is less than instrumental width
        corr_del_v2_sq  = 0.0

    return corr_del_v1_sq, corr_del_v2_sq

def plot_absorber(spectra, absorber, zabs, show_error=False, xlabel='obs wave (ang)', ylabel='residual', title='QSO', plot_filename=None):
    """
    Saves a plot of spectra with absorber(s) (full spectrum + zoomed version)
    in the current working directory.

    Args:
        spectra (object): spectra class, output of QSOSpecRead()
        absorber (str): Type of absorber, e.g., 'MgII', 'CIV'.
        zabs (list, array, or Table): Absorber redshifts, or a Table with 'Z_ABS' and 'GAUSS_FIT' columns.
        show_error (bool): if error bars should be shown (default False)
        xlabel (str): The label for the x-axis. Default is 'obs wave (ang)'.
        ylabel (str): The label for the y-axis. Default is 'residual'.
        title (str): The super title of the plot. Default is 'QSO'.
        plot_filename (str): If provided, will save the plot to the given filename.
    """

    lam, residual, error = spectra.wavelength, spectra.flux, spectra.error
    # If zabs is a Table or structured array, extract redshifts and fit parameters
    if isinstance(zabs, (Table, np.ndarray)) and ('Z_ABS' in zabs.colnames or 'Z_ABS' in zabs.dtype.names):
        redshifts = zabs['Z_ABS']
        fit_params = zabs['GAUSS_FIT']
    else:
        redshifts = zabs
        fit_params = None

    if isinstance(redshifts, float):
        redshifts = [redshifts]

    num_absorbers = len(redshifts)

    # Create a grid with 2 rows: 1 for the main plot and 1 for zoomed plots
    fig = plt.figure(figsize=(13.5, 8))
    fig.subplots_adjust(hspace=0.15, wspace=0.15)  # Adjust space between plots

    # Super title for the entire figure
    fig.suptitle(title, fontsize=16)

    # Create the main plot in the first row
    ax_main = plt.subplot2grid((2, num_absorbers), (0, 0), colspan=num_absorbers)
    ax_main.plot(lam, residual, ls='-', lw=1.5, label='residual')
    if show_error:
        ax_main.plot(lam, error, ls='-', lw=1.5, label='error')
    ax_main.set_xlim(3800, 9200)
    ax_main.legend()

    # Determine the absorber line labels
    if absorber == 'MgII':
        l1, l2 = 'MgII_2796', 'MgII_2803'
    elif absorber == 'CIV':
        l1, l2 = 'CIV_1548', 'CIV_1550'
    else:
        raise ValueError(f"Unsupported absorber type: {absorber}")

    # Plot vertical lines for the absorber lines in the main plot
    for z in redshifts:
        x1, x2 = lines[l1] * (1 + z), lines[l2] * (1 + z)
        ax_main.axvline(x=x1, color='r', ls='--')
        ax_main.axvline(x=x2, color='r', ls='--')

    ax_main.set_xlabel(xlabel)
    ax_main.set_ylabel(ylabel)
    ax_main.grid(True)
    ax_main.set_ylim(-1, 2)

    # Add subplots for zoomed-in regions in the second row
    sep = 25  # Set separation for zoomed plot ranges

    for idx, z in enumerate(redshifts):
        shift_z = 1 + z
        ax_zoom = plt.subplot2grid((2, num_absorbers), (1, idx))
        x1, x2 = lines[l1] * shift_z, lines[l2] * shift_z
        mask = (lam > x1 - sep) & (lam < x2 + sep)  # Define zoom range around the lines
        if not show_error:
            ax_zoom.plot(lam[mask], residual[mask], ls='-', lw=1.5, label='data')
        else:
            ax_zoom.errorbar(lam[mask], residual[mask], yerr=error[mask], marker='o', color='C0', markersize=6, label='data')
        ax_zoom.axvline(x=x1, color='r', ls='--')
        ax_zoom.axvline(x=x2, color='r', ls='--')
        ax_zoom.set_xlim([x1 - sep, x2 + sep])

        # Determine appropriate y-limits for the subplot based on data
        y_min, y_max = residual[mask].min(), residual[mask].max()
        y_margin = 0.2 * (y_max - y_min)  # Add a margin for better visibility
        ax_zoom.set_ylim(y_min - y_margin, y_max + y_margin)

        ax_zoom.set_title(f'{absorber} at z={z:.3f}')
        ax_zoom.grid(True)
        ax_zoom.set_xlabel(xlabel)
        ax_zoom.set_ylabel(ylabel)

        # Add Gaussian fit
        if fit_params is not None:
            params = fit_params[idx]
            # Adjust fit parameters for the redshift
            # Plot the Gaussian fit
            lam_fit = np.linspace(x1 - sep, x2 + sep, 1000)
            fit_curve = double_gaussian(lam_fit, params[0], shift_z * params[1], shift_z * params[2],
            params[3], shift_z * params[4], shift_z * params[5])
            ax_zoom.plot(lam_fit, fit_curve, 'r-', label='Gaussian Fit')
        ax_zoom.legend()

    # Use tight_layout to ensure there are no overlaps
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Reserve space for suptitle

    # Save or display the plot
    if plot_filename is not None:
        # Get the current working directory
        current_dir = os.getcwd()

        # Define the full path for the plot
        plot_path = os.path.join(current_dir, plot_filename)

        # Save the plot
        plt.savefig(plot_path)
        plt.close()

        print(f"Plot saved as {plot_path}")
    else:
        plt.show()


def read_nqso_from_header(file_path, hdu_name='METADATA'):
    """
    Read the NAXIS2 value from the header of a specified HDU in a FITS file.

    Parameters:
    - file_path: str, path to the FITS file.
    - hdu_name: str, name of the HDU from which to read NAXIS1 (default: 'METADATA').

    Returns:
    - naxis2_value: int, value of NAXIS2 from the specified HDU header.
    """
    # Check if the file exists
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The FITS file {file_path} does not exist.")

    # Open the FITS file in read-only mode and load headers only
    with fits.open(file_path, mode='readonly') as hdul:
        # Attempt to access the specified HDU by name
        try:
            # Load only the header of the specified HDU
            header = hdul[hdu_name].header

            # Read the NAXIS1 value from the header
            naxis2_value = header.get('NAXIS2', None)

            if naxis2_value is None:
                raise KeyError(f"NAXIS2 not found in the '{hdu_name}' HDU header.")
            return naxis2_value

        except KeyError:
            raise ValueError(f"No '{hdu_name}' HDU found in {file_path}.")
