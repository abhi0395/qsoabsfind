import time
#import logging
import numpy as np
from .config import lines, amplitude_dict
import matplotlib.pyplot as plt
import os

# Configure logging
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Set the logging level for Matplotlib to WARNING to suppress DEBUG messages
#logging.getLogger('matplotlib').setLevel(logging.WARNING)

def elapsed(start, msg):
    """
    Prints the elapsed time since `start`.

    Parameters:
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

    Parameters:
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

    Parameters:
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
        raise ValueError(f"Unsupported absorber type for specific parameters: {absorber}")
    if log:
        lam_ker = np.arange(np.log10(lam_ker_start), np.log10(lam_ker_end), 0.0001) #SDSS-like wavelength resolution
        lam_ker = 10**lam_ker
    else:
        lam_ker = np.arange(lam_ker_start, lam_ker_end, 0.8) # DESI-like wavelength resolution

    if len(lam_ker)>len(residual_arr_after_mask):
        lam_ker = lam_ker[0: len(residual_arr_after_mask)]

    gauss_kernel = gauss_two_lines_kernel(lam_ker, a=ker_parm)

    if index is not None:
        save_plot(lam_ker, gauss_kernel, f'kernel_{index}.png', 'wave', 'flux', f'{index}')

    result = np.convolve(gauss_kernel, residual_arr_after_mask, mode='same')

    #check if input and output array size are same
    bad_conv = validate_sizes(result, residual_arr_after_mask, index)
    if bad_conv == 1:
        print(f"ERROR: Size mismatch detected in spec_index {index}")
    return result

def double_gaussian(x, amp1, mean1, sigma1, amp2, mean2, sigma2):
    """
    Generates a double Gaussian function to fit absorption features in a given spectrum.

    Parameters:
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
    Defines the fitting function to fit a single absorption line with a gaussian profile.

    Parameters:
    x (numpy.ndarray): Wavelength points where the user wants to fit the model.
    params (list or numpy.ndarray): Array of parameters [amp, mean, sigma].

    Returns:
    numpy.ndarray: The fitting function values.
    """
    amp, mean, sigma = params
    return -amp * np.exp(-((x - mean) / sigma) ** 2 / 2) + 1

def save_plot(x, y, plot_filename='qsoabsfind_plot.png', xlabel='X-axis', ylabel='Y-axis', title='Plot Title'):
    """
    Saves a plot of x vs y in the current working directory. If y is a list of arrays, each will be plotted.

    Parameters:
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

def validate_sizes(conv_arr, unmsk_residual, spec_index):
    """
    Validate that all arrays have the same size.

    Parameters:
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


def plot_absorber(lam, residual, absorber, zabs, xlabel='obs wave (ang)', ylabel='residual', title='QSO', plot_filename=None):
    """
    Saves a plot of spectra with absorber in the current working directory.

    Parameters:
    lam (array-like): observed wavelength.
    residual (array-like): residual.
    absorber (str): e.g. MgII, CIV
    zabs(list): list of absorbers
    xlabel (str): The label for the x-axis. Default is 'X-axis'.
    ylabel (str): The label for the y-axis. Default is 'Y-axis'.
    title (str): The title of the plot. Default is 'Plot Title'.
    plot_filename (str): if provided, will save the plot
    """
    # Create the plot
    plt.figure(figsize=(12,4))
    plt.plot(lam, residual, ls='-', lw=1.5)
    if absorber=='MgII':
        l1, l2 = 'MgII_2796', 'MgII_2803'
    if absorber=='CIV':
        l1, l2 = 'CIV_1548', 'CIV_1550'
    for z in zabs:
        x1, x2 = lines[l1]*(1+z), lines[l2]*(1+z)
        plt.axvline(x = x1, color='r', ls='--')
        plt.axvline(x = x2, color='r', ls='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.ylim(-1, 2)

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
