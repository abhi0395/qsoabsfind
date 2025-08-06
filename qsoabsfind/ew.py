"""
This script contains a function to fit a given absorption profile with a double gaussian and measure equivalent widths.
"""

import numpy as np
from numba import njit
from scipy.optimize import curve_fit
from .utils import double_gaussian
from .absorberutils import redshift_estimate
from .config import load_constants

# Constants
constants = load_constants()
lines = constants.lines
doublet_keys = constants.doublet_keys
oscillator_params = constants.oscillator_parameters

def return_line_centers(use_kernel):
    """
    Return line centers for a given absorber

    Args:
        use_kerne (str): absorber (e.g. MgII, CIV, OVI, NV, AlIII, SiIV, FeII)

    Returns:
        line centers (floats)
    """
    if use_kernel not in doublet_keys:
        raise ValueError(f"Unsupported kernel type. Use {doublet_keys.keys()}")
    else:
        line_centre1, line_centre2  = lines[doublet_keys[use_kernel][0]], lines[doublet_keys[use_kernel][1]]
    return line_centre1, line_centre2

# Example usage within double_curve_fit
def double_curve_fit(index, fun_to_run, lam_fit_range, nmf_resi_fit, error_fit, bounds, init_cond, maxefv):
    """
    Fits a double Gaussian function to the provided data.

    Args:
        index (int): Index of the spectrum being fitted.
        fun_to_run (callable): The fitting function.
        lam_fit_range (numpy.ndarray): Wavelength range for the fitting.
        nmf_resi_fit (numpy.ndarray): Residual array for the fitting.
        error_fit (numpy.ndarray): Error array for the fitting.
        bounds (tuple): Bounds for the fitting parameters.
        init_cond (list or numpy.ndarray): Initial conditions for the fitting parameters.
        maxefv (int): Maximum number of iterations for the fitting algorithm.

    Returns:
        tuple: Contains the following elements:
            - save_param_array (numpy.ndarray): Fitted parameters.
            - save_param_error (numpy.ndarray): Errors of the fitted parameters.
            - EW_first (float): Equivalent width of the first Gaussian.
            - EW_second (float): Equivalent width of the second Gaussian.
            - EW_total (float): Total equivalent width of both Gaussians.
    """
    nparm = len(init_cond)
    save_param_array = np.zeros(nparm)
    save_param_error = np.zeros(nparm)
    save_param_cov = np.zeros((nparm, nparm))
    EW_first = np.nan
    EW_second = np.nan
    EW_total = np.nan
    if bounds is None:
        bounds = (-np.inf, np.inf)
    try:
        popt, pcov = curve_fit(
            fun_to_run, lam_fit_range, nmf_resi_fit,
            bounds=bounds, sigma=error_fit, p0=init_cond,
            maxfev=maxefv, absolute_sigma=True,
            ftol=1e-4, xtol=1e-4
        )
        EW_first = popt[0] * np.sqrt(np.pi * 2 * popt[2] ** 2)
        EW_second = popt[3] * np.sqrt(np.pi * 2 * popt[5] ** 2)
        EW_total = EW_first + EW_second

        save_param_array = popt
        save_param_error = np.sqrt(np.diag(pcov))
        save_param_cov = pcov
    except (RuntimeError, ValueError, TypeError) as e:
        if isinstance(e, TypeError):
            print(f'NMF resi fit size = {nmf_resi_fit.size}')
            print(f'\nIn Main Gaussian script: In double Gaussian fitting: Spec Index = {index} has issues, Check this please\n')
        save_param_array[:] = np.nan
        save_param_error[:] = np.nan
        save_param_cov[:] = np.nan

    return save_param_array, save_param_error, EW_first, EW_second, EW_total, save_param_cov

@njit
def calculate_ew_errors(popt, perr):
    """
    Calculate the errors in the equivalent widths (EW) using the errors in
    the optimized parameters.

    Args:
        popt (numpy.ndarray): Optimized parameters from the curve fitting.
        perr (numpy.ndarray): Errors of the optimized parameters from the curve fitting.

    Returns:
        tuple: Contains the following elements:
            - EW1_error (float): Error in the equivalent width of the first Gaussian.
            - EW2_error (float): Error in the equivalent width of the second Gaussian.
            - EW_total_error (float): Total error in the equivalent width of both Gaussians.
    """
    amp1, mean1, sigma1, amp2, mean2, sigma2 = popt
    amp1_err, mean1_err, sigma1_err, amp2_err, mean2_err, sigma2_err = perr

    EW1 = amp1 * np.sqrt(np.pi * 2 * sigma1 ** 2)
    EW2 = amp2 * np.sqrt(np.pi * 2 * sigma2 ** 2)

    # using correlation between parameters
    EW1_error = EW1 * np.sqrt((amp1_err / amp1) ** 2 + (sigma1_err / sigma1) ** 2 - 2 * amp1_err * sigma1_err / (amp1 * sigma1))
    EW2_error = EW2 * np.sqrt((amp2_err / amp2) ** 2 + (sigma2_err / sigma2) ** 2 - 2 * amp2_err * sigma2_err / (amp2 * sigma2))

    EW_total_error = np.sqrt(EW1_error ** 2 + EW2_error ** 2)

    return EW1_error, EW2_error, EW_total_error

def full_covariance_ew_errors(popt, pcov):
    """
    With full covariance matrix, calculate the errors in the equivalent
    widths (EW) using the errors in the optimized parameters.

    Args:
        popt (numpy.ndarray): Optimized parameters from the curve fitting.
        pcov (numpy.ndarray): Covariance matrix from the curve fitting.

    Returns:
        tuple: Contains the following elements:
            - EW1_error (float): Error in the equivalent width of the first Gaussian.
            - EW2_error (float): Error in the equivalent width of the second Gaussian.
            - EW_total_error (float): Total error in the equivalent width of both Gaussians.
    """
    # Extract optimized parameters
    amp1, mean1, sigma1, amp2, mean2, sigma2 = popt

    # Calculate the partial derivatives of EW1 and EW2 with respect to the parameters
    dEW1_damp1 = np.sqrt(2 * np.pi) * sigma1
    dEW1_dsigma1 = amp1 * np.sqrt(2 * np.pi)

    dEW2_damp2 = np.sqrt(2 * np.pi) * sigma2
    dEW2_dsigma2 = amp2 * np.sqrt(2 * np.pi)

    # Derivatives arrays for covariance calculation
    jacobian_EW1 = np.array([dEW1_damp1, 0, dEW1_dsigma1, 0, 0, 0])
    jacobian_EW2 = np.array([0, 0, 0, dEW2_damp2, 0, dEW2_dsigma2])

    # Calculate the variance (square of the error) using the full covariance matrix
    EW1_var = np.dot(jacobian_EW1, np.dot(pcov, jacobian_EW1.T))
    EW2_var = np.dot(jacobian_EW2, np.dot(pcov, jacobian_EW2.T))

    # The square root of the variance gives the error
    EW1_error = np.sqrt(EW1_var)
    EW2_error = np.sqrt(EW2_var)

    # Total EW error, including cross-terms
    cross_term = 2 * np.dot(jacobian_EW1, np.dot(pcov, jacobian_EW2.T))
    EW_total_error = np.sqrt(EW1_var + EW2_var + cross_term)

    return EW1_error, EW2_error, EW_total_error

def find_z_from_minimum(wavelength, residual, line_rest, z_guess, window=5):
    """Find better z estimate using flux minimum near expected line center."""
    lam_expected = line_rest * (1 + z_guess)
    delta = window * (wavelength[1] - wavelength[0])
    mask = (wavelength > lam_expected - delta) & (wavelength < lam_expected + delta)

    if np.any(mask):
        idx_min = np.nanargmin(residual[mask])
        lam_min = wavelength[mask][idx_min]
        return lam_min / line_rest - 1
    else:
        return z_guess  # fallback

def initialize_output_arrays(size_array, nparm):
    """
    Function to return arrays for storing Absorber fit parameters

    Args:
        size_array (int): number of redshifts
        nparm (int): number of Gaussian parameters
    Returns:
        tuple of arrays
    """
    return (
        np.zeros((size_array, nparm)),  # fitting_param_for_spectrum
        np.zeros((size_array, nparm)),  # fitting_param_std_for_spectrum
        np.zeros((size_array, nparm, nparm)),  # fitting_param_pcov_for_spectrum
        np.zeros(size_array, dtype='float32'),  # EW_first_line
        np.zeros(size_array, dtype='float32'),  # EW_second_line
        np.zeros(size_array, dtype='float32'),  # EW_first_line_error
        np.zeros(size_array, dtype='float32'),  # EW_second_line_error
        np.zeros(size_array, dtype='float32'),  # EW_total
        np.zeros(size_array, dtype='float32'),  # EW_total_error
        np.zeros(size_array, dtype='float32')   # z_abs_err
    )

def get_rest_frame_values(wavelength, flux, error, z, ix0, ix1):
    """
    Get rest-frame arrays from observed frame

    Args:
        wavelength (array): Observed wavelength
        flux (array): Observed residual
        error (array): corresponding error array
        z (float): absorber redshift
        ix0 (float): start wavelength in rest-frame
        ix1 (float): end wavelength in rest-frame
    Returns:
        wavelength, flux and error
    """
    lam_rest = wavelength / (1 + z)
    lam_ind = np.where((lam_rest >= ix0) & (lam_rest <= ix1))[0]
    return lam_rest[lam_ind], flux[lam_ind], error[lam_ind]

def get_observed_frame_values(fit_params, redshift):
    """
    Get observed-frame parameters

    Args:
        fit_params (np.ndarray): rest-frame Gaussian parameters
        redshift (float): absorber redshift
    Returns:
        observed wavelengths for line1, line2, and sigmas
    """
    l1_obs = fit_params[1] * (1 + redshift)
    l2_obs = fit_params[4] * (1 + redshift)
    sig1_obs = fit_params[2] * (1 + redshift)
    sig2_obs = fit_params[5] * (1 + redshift)
    return l1_obs, l2_obs, sig1_obs, sig2_obs

def measure_absorber_properties_double_gaussian(index, wavelength, flux, error, absorber_redshift, bound, use_kernel, d_pix, num_iter=1000, use_covariance=False):

    """
    Measures the properties of each potential absorber by fitting a double
    Gaussian to the absorption feature and measuring the equivalent width (EW)
    and errors of absorption lines.

    Args:
        index (int): Index of the spectrum being fitted.
        wavelength (numpy.ndarray): Array containing common rest frame quasar wavelength.
        flux (numpy.ndarray): Matrix containing the residual flux.
        error (numpy.ndarray): Error array corresponding to the flux.
        absorber_redshift (list): List of potential absorbers identified previously.
        bound (tuple): Bounds for the fitting parameters.
        use_kernel (str, optional): Kernel type ('MgII, FeII, CIV, NV, OVI, SiIV, AlIII).
        d_pix (float, optional): wavelength pixel for tolerance
        num_iter (int): similar to maxefv option in curve_fit, maximum number of iterations for function evalution (default 1000)
        use_covariance (bool): if want to use full covariance of scipy curvey_fit for EW error calculation (default is False)

    Returns:
        tuple: Contains the following elements:
            - z_abs_array (numpy.ndarray): Array of absorber redshifts.
            - fitting_param_for_spectrum (numpy.ndarray): Array of fitting parameters for double Gaussian.
            - fitting_param_std_for_spectrum (numpy.ndarray): Array of errors for fitting parameters.
            - EW_first_line (numpy.ndarray): Mean equivalent width of the first line.
            - EW_second_line (numpy.ndarray): Mean equivalent width of the second line.
            - EW_total (numpy.ndarray): Mean total equivalent width of both lines.
            - EW_first_line_error (numpy.ndarray): Error in the equivalent width of the first line.
            - EW_second_line_error (numpy.ndarray): Error in the equivalent width of the second line.
            - EW_total_error (numpy.ndarray): Total error in the equivalent width of both lines.
    """

    z_abs_array = np.array(absorber_redshift)
    size_array = z_abs_array.size
    nparm = 6

    (fitting_param, fitting_param_std, fitting_param_pcov,
     EW1, EW2, EW1_err, EW2_err, EW_total, EW_total_err, z_abs_err) = initialize_output_arrays(size_array, nparm)

    line1, line2 = return_line_centers(use_kernel)
    amp_ratio = oscillator_params[f'{use_kernel}_f2'] / oscillator_params[f'{use_kernel}_f1']
    sigma = d_pix * 15
    ix0, ix1 = line1 - sigma, line2 + sigma
    pixel_width = 5

    if size_array == 0:
        return z_abs_array, fitting_param, fitting_param_std, EW1, EW2, EW_total, EW1_err, EW2_err, EW_total_err

    for k in range(size_array):
        np.random.seed(int(z_abs_array[k] * 1e6) % 2**32)

        z1 = find_z_from_minimum(wavelength, flux, line1, z_abs_array[k], window=pixel_width)
        z2 = find_z_from_minimum(wavelength, flux, line2, z_abs_array[k], window=pixel_width)
        z_abs_array[k] = (line1 * z1 + line2 * z2) / (line1 + line1)

        lam_fit, nmf_resi, error_flux = get_rest_frame_values(wavelength, flux, error, z_abs_array[k], ix0, ix1)

        if nmf_resi.size > 0 and not np.all(np.isnan(nmf_resi)):
            amp1 = max(0.05, 1 - np.nanmin(nmf_resi))
            amp2 = min(0.95, amp_ratio * amp1)
            sigma1 = np.random.uniform(bound[0][2], bound[1][2])
            sigma2 = np.random.uniform(bound[0][5], bound[1][5])
            init_cond = [amp1, line1, sigma1, amp2, line2, sigma2]

            fit, fit_std, ew1, ew2, ew_total, _ = double_curve_fit(
                index, double_gaussian, lam_fit, nmf_resi, error_flux, bounds=bound, init_cond=init_cond, maxefv=num_iter)

            fitting_param[k] = fit
            fitting_param_std[k] = fit_std
            EW1[k], EW2[k], EW_total[k] = ew1, ew2, ew_total

            l1_obs, l2_obs, sig1_obs, sig2_obs = get_observed_frame_values(fit, z_abs_array[k])
            obs_init_cond = [amp1, l1_obs, sig1_obs, amp2, l2_obs, sig2_obs]

            obs_fit, obs_fit_std, *_ = double_curve_fit(
                index, double_gaussian, lam_fit * (1 + z_abs_array[k]), nmf_resi, error_flux,
                bounds=None, init_cond=obs_init_cond, maxefv=num_iter)

            z_abs_array[k], z_abs_err[k] = redshift_estimate(
                obs_fit[1], obs_fit[4], obs_fit_std[1], obs_fit_std[4], line1, line2)

            lam_fit, nmf_resi, error_flux = get_rest_frame_values(wavelength, flux, error, z_abs_array[k], ix0, ix1)

            fit, fit_std, ew1, ew2, ew_total, pcov = double_curve_fit(
                index, double_gaussian, lam_fit, nmf_resi, error_flux, bounds=bound, init_cond=init_cond, maxefv=num_iter)

            fitting_param[k] = fit
            fitting_param_std[k] = fit_std
            fitting_param_pcov[k] = pcov
            EW1[k], EW2[k], EW_total[k] = ew1, ew2, ew_total

            l1_obs, l2_obs = fit[1] * (1 + z_abs_array[k]), fit[4] * (1 + z_abs_array[k])
            std_l1, std_l2 = fit_std[1] * (1 + z_abs_array[k]), fit_std[4] * (1 + z_abs_array[k])
            z_abs_array[k], z_abs_err[k] = redshift_estimate(l1_obs, l2_obs, std_l1, std_l2, line1, line2)

            if not use_covariance:
                EW1_err[k], EW2_err[k], EW_total_err[k] = calculate_ew_errors(fit, fit_std)
            else:
                EW1_err[k], EW2_err[k], EW_total_err[k] = full_covariance_ew_errors(fit, pcov)

            if np.all(np.isnan([ew1, ew2, ew_total])):
                fitting_param[k] = 0
                fitting_param_std[k] = 0
                fitting_param_pcov[k] = 0
                EW1[k], EW2[k], EW_total[k] = 0, 0, 0
                EW1_err[k], EW2_err[k], EW_total_err[k], z_abs_err[k] = 0, 0, 0, 0
        else:
            fitting_param[k] = 0
            fitting_param_std[k] = 0
            fitting_param_pcov[k] = 0
            EW1[k], EW2[k], EW_total[k] = 0, 0, 0
            EW1_err[k], EW2_err[k], EW_total_err[k], z_abs_err[k] = 0, 0, 0, 0

    return (
        z_abs_array, z_abs_err, fitting_param, fitting_param_std,
        EW1, EW2, EW_total, EW1_err, EW2_err, EW_total_err
    )
