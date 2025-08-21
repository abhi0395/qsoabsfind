"""
This script contains some utility functions.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits
from astropy.table import Table, Row
import re
from importlib.metadata import version, PackageNotFoundError

# Constants
from .constants import lines, oscillator_parameters, speed_of_light, doublet_keys, amplitude_dict

def get_package_versions():
    """
    Get the versions of qsoabsfind and other relevant packages.

    Returns:
        dict: A dictionary containing the versions of the packages.
    """
    packages = ['qsoabsfind', 'numpy', 'astropy', 'scipy', 'numba', 'matplotlib']
    versions = {}
    for pkg in packages:
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            versions[pkg] = 'not installed'
    return versions

def get_all_extnames(filename):
    """Get list of all HDU extension names in a FITS file"""

    with fits.open(filename) as hdul:
        extnames = []
        for i, hdu in enumerate(hdul):
            name = hdu.name if hdu.name else f"HDU_{i}"
            extnames.append((i, name, type(hdu).__name__))

    return extnames


def update_header(args, user_constants):
    """Add search parameters and package versions to headers.

    Updates FITS header with search parameters from command-line arguments and
    user-defined constants, along with relevant package version information for
    reproducibility.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing search
            parameters and configuration options.
        user_constants (object): Object containing user-defined constants and
            configuration attributes used in the search/analysis.

    Returns:
        astropy.io.fits.Header: Updated FITS header containing search parameters,
            constants, and package version information.
    """
    # Prepare headers
    headers = {}
    for header in args.headers:
        key, value = header.split('=')
        headers[key] = {"value": value, "comment": ""}

    headers.update({
        'ABSORBER': {"value": args.absorber, "comment": 'Absorber name'},
        'KERWIDTH': {"value": str(user_constants.search_parameters["ker_width_pixels"]), "comment": 'Kernel width in pixels (ker_width_pixels)'},
        'COEFFSIG': {"value": user_constants.search_parameters["coeff_sigma"], "comment": 'sigma threshold (coeff_sigma)'},
        'MULTRE': {"value": user_constants.search_parameters["mult_resi"], "comment": 'Multiplicative factor for residuals (mult_resi)'},
        'D_PIX': {"value": user_constants.search_parameters["d_pix"], "comment": 'tolerance for line separation (in Ang) (d_pix)'},
        'PM_PIXEL': {"value": user_constants.search_parameters["pm_pixel"], "comment": 'N_Pixel for noise estimation (pm_pixel)'},
        'SN_LINE1': {"value": user_constants.search_parameters["sn_line1"], "comment": 'S/N threshold for first line (sn_line1)'},
        'SN_LINE2': {"value": user_constants.search_parameters["sn_line2"], "comment": 'S/N threshold for second line (sn_line2)'},
        'EWCOVAR': {"value": user_constants.search_parameters["use_covariance"], "comment": 'Use covariance for EW error (use_covariance)'},
        'LOGWAVE': {"value": user_constants.search_parameters["logwave"], "comment": 'Use log wavelength scaling (logwave)'},
        'LAM_ESEP': {"value": user_constants.search_parameters["lam_edge_sep"], "comment": 'lambda edges to avoid edges (lam_edge_sep)'},
        'BLUE_LAM': {"value": user_constants.search_parameters["lam_blue"], "comment": 'blue end of quasar-rest frame (Ang) for search'},
        'RED_LAM': {"value": user_constants.search_parameters["lam_red"], "comment": 'red end of quasar-rest frame (Ang) for search'},
        'CONTERR': {"value": user_constants.search_parameters["continuum_error_frac"], "comment": 'fractional error in continuum normalization'},
        'CONFLEV': {"value": user_constants.search_parameters["conf_level"], "comment": 'minimum confidence level for selection'},
    })

    return headers

def parse_qso_sequence(qso_sequence):
    """
    Parse a bash-like sequence or a single integer to generate QSO indices.

    Args:
        qso_sequence (str or int): Bash-like sequence (e.g., '1-1000', '1-1000:10') or an integer.

    Returns:
        numpy.array: Array of QSO indices.
    """
    if isinstance(qso_sequence, int):
        return np.arange(qso_sequence)

    # Handle string input
    if isinstance(qso_sequence, str):
        if qso_sequence.isdigit():
            return np.arange(int(qso_sequence))

        match = re.match(r"(\d+)-(\d+)(?::(\d+))?", qso_sequence)
        if match:
            start, end, step = match.groups()
            start, end = int(start), int(end)
            step = int(step) if step else 1
            return np.arange(start, end + 1, step)

    # If none of the conditions matched, raise an error
    raise ValueError(f"Invalid QSO sequence format: '{qso_sequence}'. Use 'start-end[:step]' or an integer.")


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

    norm_constant = -1

    return norm_constant * (-a1 * np.exp(-((x - a[1]) / a[2]) ** 2 / 2) - a2 * np.exp(-((x - a[4]) / a[5]) ** 2 / 2)) * 0.5 + 1

def compute_doublet_amplitudes(A1_input, f1, f2):
    """
    Computes amplitudes for the first and second lines of a doublet
    based on the user-defined A1_input and oscillator strengths f1 and f2.

    Ensures that both amplitudes remain <= 1.

    Args:
        A1_input (float): Desired amplitude of the stronger line (usually <= 1).
        f1 (float): Oscillator strength of the first line.
        f2 (float): Oscillator strength of the second line.

    Returns:
        (A1, A2): Tuple of amplitudes for line1 and line2
    """

    # Normalize f1 and f2 such that max(A1, A2) = A1_input if possible
    ratio = f2 / f1
    A2 = A1_input * ratio

    # If A2 exceeds 1, we need to scale both amplitudes down
    if A2 > 1.0:
        scale = 1.0 / A2
        A1 = A1_input * scale
        A2 = 1.0
    else:
        A1 = A1_input

    return A1, A2


def convolution_fun(absorber, residual_arr_after_mask, width, log, wave_res, index, f1, f2):
    """
    Convolves the spectrum with a Gaussian kernel.

    Args:
        absorber (str): Type of absorber (e.g., MgII, CIV, OVI, NV, SiIV, AlIII, FeII).
        residual_arr_after_mask (numpy.ndarray): Final residual array after masking.
        width (float): The width of the Gaussian kernel (decide base dupon width of real absorption feature).
        log (bool): if log bins should be used for wavelength
        wave_res (float): wavelength pixel size (SDSS: 0.0001 on log scale, DESI: 0.8 on linear scale)
        index (int): QSO index
        f1 (float): Oscillator strength of the first line.
        f2 (float): Oscillator strength of the second line.

    Returns:
        numpy.ndarray: The convolved residual array.
    """
    if absorber not in amplitude_dict:
        raise ValueError(f"Unsupported absorber type. Available types are: {list(amplitude_dict.keys())}")

    A_main = amplitude_dict[absorber]
    A_main, A_secondary = compute_doublet_amplitudes(A_main, f1, f2)
    ct = 10
    # extract lambdas for the doublet
    lambda1, lambda2 = lines[doublet_keys[absorber][0]], lines[doublet_keys[absorber][1]]

    ker_parm = np.array([A_main, lambda1, width, A_secondary, lambda2, width])
    lam_ker_start = lambda1 - ct * width # +/- 10sigma , #rest-frame
    lam_ker_end = lambda2 + ct * width

    if log:
        lam_ker = np.arange(np.log10(lam_ker_start), np.log10(lam_ker_end)+wave_res, wave_res) #SDSS-like wavelength resolution
        lam_ker = 10**lam_ker
    else:
        lam_ker = np.arange(lam_ker_start, lam_ker_end+wave_res, wave_res) # DESI-like wavelength resolution

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


def modify_units(col_name, col):
    """
    Modify the unit of a column based on the column name.

    Args:
        col_name (str): The name of the column.
        col (Column): The column object.

    Returns:
        str: The modified unit if conditions are met, otherwise the original unit.
    """
    if 'EW' in col_name.upper():
        return 'Angstrom'
    elif 'VDISP' in col_name.upper():
        return 'km s-1'
    elif '10N' in col_name.upper():
        return 'cm-2'
    else:
        return str(col.unit) if col.unit is not None else None

def numeric_key(filename):
    """Extract first integer from filename for sorting.

    Extracts the first sequence of digits found in a filename to use as a numeric
    sorting key. Files without numbers are placed at the end of the sort order.

    Args:
        filename (str): Filename or path from which to extract the numeric key.

    Returns:
        int or float: First integer found in the filename, or float('inf') if no
            number is present (ensuring numberless files sort last).

    Examples:
        >>> numeric_key("file_123_data.txt")
        123
        >>> numeric_key("report_42.pdf")
        42
        >>> numeric_key("no_numbers_here.txt")
        inf
        >>> sorted(["file3.txt", "file20.txt", "file1.txt"], key=numeric_key)
        ['file1.txt', 'file3.txt', 'file20.txt']
    """
    import re
    match = re.search(r'\d+', filename)
    return int(match.group()) if match else float('inf')  # put no-number files at end


def combine_fits_files(directory, output_file):
    """
    Combines data from several FITS files in a directory into a single FITS file.

    Note:
        This function assumes that all FITS files have the same HDU structure and that all HDUs are Tables. Any column with 'EW' in its name will have its unit set to Angstrom. The `INDEX_SPEC` column is not concatenated. The primary HDU from the first FITS file is copied to the final combined file. So make sure that directory contains only qsoabsfind output files.

    Args:
        directory (str): Path to the directory containing the FITS files.
        output_file (str): Path to the output FITS file.

    Returns:
        None: The combined FITS file is saved to the specified output path.
    """
    from astropy.table import vstack
    # Initialize a dictionary to store tables for each HDU
    combined_tables = {}
    primary_hdu = None
    formats = {}
    # Loop through each file in the directory
    for i, file_name in enumerate(sorted(os.listdir(directory), key=numeric_key)):
        if file_name.endswith('.fits'):
            file_path = os.path.join(directory, file_name)
            print(f"Processing file: {file_path}")
            with fits.open(file_path) as hdul:
                if primary_hdu is None:
                    # Copy the primary HDU from the first file
                    primary_hdu = fits.PrimaryHDU(header=hdul[0].header)
                    print("Primary HDU copied from the first file.")
                    for hdu in hdul[1:]:
                        hdu_name = hdu.name
                        formats[hdu_name] = {}
                        for col in hdu.columns:
                            formats[hdu.name][col.name] = col.format
                for hdu in hdul:
                    if isinstance(hdu, fits.BinTableHDU):
                        hdu_name = hdu.name
                        table = Table.read(file_path, hdu=hdu_name)

                        # Remove the INDEX_SPEC column if it exists
                        if 'INDEX_SPEC' in table.colnames:
                            table.remove_column('INDEX_SPEC')
                            print(f"Removed 'INDEX_SPEC' column from HDU '{hdu_name}'.")

                        if hdu_name in combined_tables:
                            combined_tables[hdu_name] = vstack([combined_tables[hdu_name], table], metadata_conflicts='silent')
                            print(f"Concatenated data to HDU '{hdu_name}'.")
                        else:
                            combined_tables[hdu_name] = table
                            print(f"Initialized HDU '{hdu_name}' with data from file {i + 1}.")

    # Create the HDUs to write to the output file
    primary_hdu.header['EXTNAME'] = 'PRIMARY'
    hdul_out = fits.HDUList([primary_hdu])

    for hdu_name, table in combined_tables.items():
        # Explicitly handle units by creating columns with modified unit information
        columns = []
        for col_name in table.colnames:
            col = table[col_name]

            # Modify the unit based on the column name
            unit = modify_units(col_name, col)

            # Create the FITS column, let the format be inferred
            fits_format = formats[hdu_name][col_name]
            columns.append(fits.Column(name=col_name, array=col.data, format=fits_format, unit=unit))
        hdu_out = fits.BinTableHDU.from_columns(columns, name=hdu_name)
        hdul_out.append(hdu_out)
        print(f"Added HDU '{hdu_name}' to the output file.")

    # Write the combined data to the output FITS file
    hdul_out.writeto(output_file, overwrite=True)
    print(f"Combined FITS file saved to {output_file}")

def match_order(arr1, arr2):
    """Matching order based on match key fot given two arrays

    Args:
        arr1 (array): First array (will be assumed to be the reference array)
        arr2 (array): Second array (target array for which the order to be matched)

    Returns:
        matching indices such that arr2[indices]=arr1

    Note:
        Raises Assertion error if sizes do not match
    """

    assert arr1.size==arr2.size

    indices = []
    for el in arr2:
        ii = np.where(arr1 == el)[0]
        indices.append(ii)
    indices = np.array(indices).flatten()
    updated_arr2 = arr2[indices]
    np.testing.assert_array_equal(arr1, updated_arr2)
    return indices


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
        print(f"ERROR: Size mismatch detected in spec_index {spec_index}")
    return bad_conv

def vel_dispersion(c1, c2, sigma1, sigma2, resolution, z, obs_wave):
    """
    Calculates and corrects velocity dispersion using Gaussian quadrature.

    Args:
        c1 (float): rest-frame fitted line center 1 (in Ang).
        c2 (float): rest-frame fitted line center 2 (in Ang).
        sigma1 (float): rest-frame fitted width 1 (in Ang).
        sigma2 (float): rest-frame fitted width 2 (in Ang).
        resolution (float or np.array): instrumental true resolution (in km/s), see note.
        z (float): redshift of absorber
        obs_wave (np.array): observed wavelength in Angstroms

    Returns:
        instrumental resolution corrected velocity dispersion in km/s

    Note:
        - resolution must be the true one, not the FWHM, usually R = lambda/delta_lambda is in FWHM unit, so first divide by 2.355 and then provide here. This is important.
    """

    v1_sig = sigma1 / c1 * speed_of_light
    v2_sig = sigma2 / c2 * speed_of_light

    lam_obs1 = (1 + z) * c1
    lam_obs2 = (1 + z) * c2

    # Get per-line instrumental sigma_v (km/s)
    if np.isscalar(resolution):
        res1 = float(resolution)
        res2 = float(resolution)
    else:
        # Interpolate instrumental sigma_v at the exact observed wavelengths
        # Assumes obs_wave is monotonic and same length as resolution.
        res1 = float(np.interp(lam_obs1, obs_wave, resolution))
        res2 = float(np.interp(lam_obs2, obs_wave, resolution))

    #Gaussian quadrature correction
    del_v1_sq = v1_sig**2 - res1**2
    del_v2_sq = v2_sig**2 - res2**2

    is_resolved1 = del_v1_sq >= 0
    is_resolved2 = del_v2_sq >= 0

    # Correct for instrumental resolution
    # Set to NaN if the fitted  width is less than rest-frame instrumental width
    # One line may resolved and one may be not, so this condition is a little relaxed
    corr_del_v1_sq = np.sqrt(del_v1_sq) if is_resolved1 else np.nan
    corr_del_v2_sq = np.sqrt(del_v2_sq) if is_resolved2 else np.nan

    return corr_del_v1_sq, corr_del_v2_sq


def plot_absorber(spectra, absorber, zabs, show_error=False, plot_filename=None, **kwargs):
    """
    Saves a plot of spectra with absorber(s) (full spectrum + zoomed version) along
    with its Gaussian fit in the current working directory or in the user-defined
    directory.

    Args:
        spectra (object): spectra class, output of QSOSpecRead()
        absorber (str): Type of absorber, e.g., MgII, CIV, OVI, NV, SiIV, AlIII, FeII
        zabs (Table, Row, dict, np.ndarray or float): Must have 'Z_ABS' and 'GAUSS_FIT' columns, if not float.
        show_error (bool): if error bars should be shown (default False)
        plot_filename (str): If provided, will save the plot to the given filename.
        **kwargs: Additional keyword arguments for matplotlib plot functions, such as:
                  xlabel (str): The label for the x-axis.
                  ylabel (str): The label for the y-axis.
                  title (str): The super title of the plot.
                  fontsize (int): Font size for the title and labels.
    """

    # Extract common plot parameters from kwargs or set to default values
    xlabel = kwargs.pop('xlabel', 'obs wave (ang)')
    ylabel = kwargs.pop('ylabel', 'residual')
    title = kwargs.pop('title', 'QSO')
    fontsize = kwargs.pop('fontsize', 16)

    lam, residual, error = spectra.wavelength, spectra.flux, spectra.error
    # If zabs is a Table or structured array, extract redshifts and fit parameters
    if isinstance(zabs, (Table, Row, dict, np.ndarray)) and ('Z_ABS' in zabs.keys() and 'GAUSS_FIT' in zabs.keys()):
        redshifts = zabs['Z_ABS']
        fit_params = zabs['GAUSS_FIT']
    else:
        redshifts = zabs
        fit_params = None

    if isinstance(redshifts, float):
        redshifts = [redshifts]
        if fit_params is not None:
            fit_params = [fit_params]

    num_absorbers = len(redshifts)

    sep = 25  # Set separation for zoomed plot ranges

    # Create a grid with 2 rows: 1 for the main plot and 1 for zoomed plots
    fig = plt.figure(figsize=(13.5, 8))
    fig.subplots_adjust(hspace=0.15, wspace=0.15)  # Adjust space between plots

    # Super title for the entire figure
    fig.suptitle(title, fontsize=fontsize)

    # Create the main plot in the first row
    ax_main = plt.subplot2grid((2, num_absorbers), (0, 0), colspan=num_absorbers)
    ax_main.plot(lam, residual, ls='-', lw=1.5, label='residual', **kwargs)
    if show_error:
        ax_main.plot(lam, error, ls='-', lw=1.5, label='error', **kwargs)
    ymask = ~np.isnan(residual)
    xmin, xmax = lam[ymask].min(), lam[ymask].max()
    ax_main.set_xlim(xmin, xmax)
    ax_main.legend(prop={'size':11})

    # Determine the absorber line labels

    if absorber not in doublet_keys:
        raise ValueError(f"Unsupported absorber type: {absorber}")
    else:
        l1, l2 = doublet_keys[absorber][0], doublet_keys[absorber][1]

    # Plot vertical lines for the absorber lines in the main plot
    for z in redshifts:
        x1, x2 = lines[l1] * (1 + z), lines[l2] * (1 + z)
        ax_main.axvline(x=x1, color='r', ls='--')
        ax_main.axvline(x=x2, color='r', ls='--')

    ax_main.set_xlabel(xlabel, fontsize=fontsize)
    ax_main.set_ylabel(ylabel, fontsize=fontsize)
    ax_main.grid(True)
    ax_main.minorticks_on()
    ax_main.set_ylim(-1, 2)
    ax_main.tick_params(axis='both', which='major', labelsize=13)
    ax_main.tick_params(axis='both', which='minor', length=2.5, width=1, color='gray')

    # Add subplots for zoomed-in regions in the second row
    for idx, z in enumerate(redshifts):
        shift_z = 1 + z
        ax_zoom = plt.subplot2grid((2, num_absorbers), (1, idx))
        x1, x2 = lines[l1] * shift_z, lines[l2] * shift_z
        mask = (lam > x1 - sep) & (lam < x2 + sep)  # Define zoom range around the lines
        if not show_error:
            ax_zoom.plot(lam[mask], residual[mask], ls='-', lw=1.5, label='data', **kwargs)
        else:
            ax_zoom.errorbar(lam[mask], residual[mask], yerr=error[mask], marker='o', color='C0', markersize=6, label='data', **kwargs)
        ax_zoom.axvline(x=x1, color='r', ls='--')
        ax_zoom.axvline(x=x2, color='r', ls='--')
        ax_zoom.set_xlim([x1 - sep, x2 + sep])

        # Determine appropriate y-limits for the subplot based on data
        y_min, y_max = max(0, np.nanmin(residual[mask])), np.nanmax(residual[mask])
        y_margin = 0.2 * (y_max - y_min)  # Add a margin for better visibility
        ax_zoom.set_ylim(y_min - y_margin, y_max + y_margin)

        ax_zoom.set_title(f'{absorber} at z={z:.3f}', fontsize=fontsize)
        ax_zoom.minorticks_on()
        ax_zoom.grid(True)
        ax_zoom.set_xlabel(xlabel, fontsize=fontsize)
        ax_zoom.set_ylabel(ylabel, fontsize=fontsize)
        ax_zoom.tick_params(axis='both', which='major', labelsize=13)
        ax_zoom.tick_params(axis='both', which='minor', length=2.5, width=1, color='gray')
        # Add Gaussian fit
        if fit_params is not None:
            params = fit_params[idx]
            # Adjust fit parameters for the redshift
            # Plot the Gaussian fit
            lam_fit = np.linspace(x1 - sep, x2 + sep, 1000)
            fit_curve = double_gaussian(
                lam_fit, params[0], shift_z * params[1], shift_z * params[2],
                params[3], shift_z * params[4], shift_z * params[5]
            )
            ax_zoom.plot(lam_fit, fit_curve, 'r-', label='Gaussian Fit', **kwargs)
        ax_zoom.legend(prop={'size':11})

    # Use tight_layout to ensure there are no overlaps
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Reserve space for suptitle

    # Save or display the plot
    if plot_filename is not None:
        # Get the current working directory
        current_dir = os.getcwd()

        # Define the full path for the plot
        plot_path = plot_filename
        if not os.path.isabs(plot_filename):
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

    Args:
        file_path: str, path to the FITS file.
        hdu_name: str, name of the HDU from which to read NAXIS1 (default: 'METADATA').

    Returns:
        naxis2_value: int, value of NAXIS2 from the specified HDU header.
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
