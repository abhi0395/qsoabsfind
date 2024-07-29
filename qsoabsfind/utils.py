import time
import numpy as np
from .config import lines, amplitude_dict

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
    a1 = -1 / (np.sqrt(np.pi * 2 * a[2] ** 2))
    a2 = 1 / (np.sqrt(np.pi * 2 * a[5] ** 2))

    return (a1 * np.exp(-((x - a[1]) / a[2]) ** 2 / 2) - a2 * np.exp(-((x - a[4]) / a[5]) ** 2 / 2)) * 0.5

def convolution_fun(absorber, residual_arr_after_mask, width, amp_ratio=0.5, log=True):
    """
    Convolves the spectrum with a Gaussian kernel.

    Parameters:
    absorber (str): Type of absorber (e.g., 'MgII', 'CIV').
    residual_arr_after_mask (numpy.ndarray): Final residual array after masking.
    width (float): The width of the Gaussian kernel (decide base dupon width of real absorption feature).
    amp_ratio (float): Amplitude ratio for the Gaussian lines (default 0.5).
    log (bool): if log bins should be used for wavelength (dlam = 0.0001, default True)

    Returns:
    numpy.ndarray: The convolved residual array.
    """
    if absorber not in amplitude_dict:
        raise ValueError(f"Unsupported absorber type. Available types are: {list(amplitude_dict.keys())}")

    A_main = amplitude_dict[absorber]
    A_secondary = A_main * amp_ratio

    if absorber == 'MgII':
        ker_parm = np.array([A_main, lines['MgII_2796'], width, A_secondary, lines['MgII_2803'], width])
        lam_ker_start = lines['MgII_2796']-5*width # +/- 5sigma , #rest-frame
        lam_ker_end = lines['MgII_2803']+5*width
    elif absorber == 'CIV':
        ker_parm = np.array([A_main, lines['CIV_1548'], width, A_secondary, lines['CIV_1550'], width])
        lam_ker_start = lines['CIV_1548']-5*width # +/- 5sigma , #rest-frame
        lam_ker_end = lines['CIV_1550']+5*width #
    else:
        raise ValueError(f"Unsupported absorber type for specific parameters: {absorber}")
    if log:
        lam_ker = np.arange(np.log10(lam_ker_start), np.log10(lam_ker_end), 0.0001) #SDSS-like wavelength resolution
    else:
        lam_ker = np.arange(lam_ker_start, lam_ker_end, 0.8) # DESI-like wavelength resolution
    gauss_kernel = gauss_two_lines_kernel(10 ** lam_ker, a=ker_parm)

    result = np.convolve(1 - residual_arr_after_mask, gauss_kernel, mode='same')

    return result

def double_gaussian(x, params):
    """
    Generates a double Gaussian function to fit absorption features in a given spectrum.

    Parameters:
    x (numpy.ndarray): List of wavelength points where the user wants to fit the model.
    params (list or numpy.ndarray): Array of six parameters:
                                    [amp1, mean1, sigma1, amp2, mean2, sigma2]

    Returns:
    numpy.ndarray: The function that fits the absorption feature using curve_fit.
    """
    amp1, mean1, sigma1, amp2, mean2, sigma2 = params
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
