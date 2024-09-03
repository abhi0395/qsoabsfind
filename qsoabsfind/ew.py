"""
This script contains a function to fit a given absorption profile with a double gaussian and measure equivalent widths.
"""

import numpy as np
from scipy.optimize import curve_fit
from .utils import double_gaussian
from .absorberutils import redshift_estimate
from .config import load_constants

constants = load_constants()
lines = constants.lines

def return_line_centers(use_kernel):
    """
    Return line centers for a given absorber

    Args:
        use_kerne (str): absorber (e.g. MgII, CIV)

    Returns:
        line centers (floats)
    """

    if use_kernel == 'MgII':
        line_centre1 = lines['MgII_2796']
        line_centre2 = lines['MgII_2803']
    elif use_kernel == 'CIV':
        line_centre1 = lines['CIV_1548']
        line_centre2 = lines['CIV_1550']
    else:
        raise ValueError("Unsupported kernel type. Use 'MgII', or 'CIV'.")

    return line_centre1, line_centre2

# Example usage within double_curve_fit
def double_curve_fit(index, fun_to_run, lam_fit_range, nmf_resi_fit, error_fit, bounds, init_cond, iter_n):
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
        iter_n (int): Maximum number of iterations for the fitting algorithm.

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
            maxfev=iter_n, absolute_sigma=True
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

def measure_absorber_properties_double_gaussian(index, wavelength, flux, error, absorber_redshift, bound, use_kernel, d_pix, use_covariance=False):
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
        use_kernel (str, optional): Kernel type ('MgII, FeII, CIV).
        d_pix (float, optional): wavelength pixel for tolerance
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

    nparm = 6 # 6 parameter double Gaussian
    fitting_param_for_spectrum = np.zeros((size_array, nparm))
    fitting_param_std_for_spectrum = np.zeros((size_array, nparm))
    fitting_param_pcov_for_spectrum = np.zeros((size_array, nparm, nparm)) #covariance matrix from curve_fit
    EW_first_line = np.zeros(size_array, dtype='float32')
    EW_second_line = np.zeros(size_array, dtype='float32')
    EW_first_line_error = np.zeros(size_array, dtype='float32')
    EW_second_line_error = np.zeros(size_array, dtype='float32')
    EW_total = np.zeros(size_array, dtype='float32')
    EW_total_error = np.zeros(size_array, dtype='float32')
    z_abs_err = np.zeros(size_array, dtype='float32')

    line_centre1, line_centre2 = return_line_centers(use_kernel)

    #defining wwavelength range for Gaussian fitting
    sigma = d_pix*15 # assuming maximum line width of d_pix * 15, can be larger/smaller, but this is a reasonable assumption.
    ix0 = line_centre1 -  sigma
    ix1 = line_centre2 +  sigma

    if size_array == 0:
        return (
            z_abs_array, fitting_param_for_spectrum, fitting_param_std_for_spectrum,
            EW_first_line, EW_second_line, EW_total,
            EW_first_line_error, EW_second_line_error, EW_total_error
        )

    #np.random.seed(1234) # for reproducibility of initital condition
    for k in range(size_array):
        absorber_rest_lam = wavelength / (1 + absorber_redshift[k]) # rest-frame conversion of wavelength
        lam_ind = np.where((absorber_rest_lam >= ix0) & (absorber_rest_lam <= ix1))[0]
        lam_fit = absorber_rest_lam[lam_ind]
        nmf_resi = flux[lam_ind]
        error_flux = error[lam_ind]
        uniform = np.random.uniform
        if nmf_resi.size > 0 and not np.all(np.isnan(nmf_resi)):
            #random initial condition
            amp_first_nmf = 1 - np.nanmin(nmf_resi)
            line_first = line_centre1
            sigma1 = uniform(bound[0][2], bound[1][2])
            sigma2 = uniform(bound[0][5], bound[1][5])
            line_second = line_centre2
            init_cond = [amp_first_nmf, line_first, sigma1, 0.54 * amp_first_nmf, line_second, sigma2]
            fitting_param_for_spectrum[k], fitting_param_std_for_spectrum[k], EW_first_line[k], EW_second_line[k], EW_total[k],_ = double_curve_fit(
                index, double_gaussian, lam_fit, nmf_resi, error_fit=error_flux, bounds=bound, init_cond=init_cond, iter_n=2000)

            fitted_l1 = fitting_param_for_spectrum[k][1]*(1+absorber_redshift[k]) # in observed frame
            fitted_l2 = fitting_param_for_spectrum[k][4]*(1+absorber_redshift[k])
            std_fitted_l1 = fitting_param_std_for_spectrum[k][1]*(1+absorber_redshift[k])
            std_fitted_l2 = fitting_param_std_for_spectrum[k][4]*(1+absorber_redshift[k])

            z_abs_array[k], z_abs_err[k] = redshift_estimate(fitted_l1, fitted_l2, std_fitted_l1, std_fitted_l2, line_centre1, line_centre2)

            #best-fit corresponding to this best redshift
            absorber_rest_lam = wavelength / (1 + z_abs_array[k]) # rest-frame conversion of wavelength
            lam_ind = np.where((absorber_rest_lam >= ix0) & (absorber_rest_lam <= ix1))[0]
            lam_fit = absorber_rest_lam[lam_ind]
            nmf_resi = flux[lam_ind]
            error_flux = error[lam_ind]

            fitting_param_for_spectrum[k], fitting_param_std_for_spectrum[k], EW_first_line[k], EW_second_line[k], EW_total[k], fitting_param_pcov_for_spectrum[k] = double_curve_fit(
                index, double_gaussian, lam_fit, nmf_resi, error_fit=error_flux, bounds=bound, init_cond=init_cond, iter_n=1000)

            fitted_l1 = fitting_param_for_spectrum[k][1]*(1+z_abs_array[k]) # in observed frame
            fitted_l2 = fitting_param_for_spectrum[k][4]*(1+z_abs_array[k])
            std_fitted_l1 = fitting_param_std_for_spectrum[k][1]*(1+z_abs_array[k])
            std_fitted_l2 = fitting_param_std_for_spectrum[k][4]*(1+z_abs_array[k])

            z_abs_array[k], z_abs_err[k] = redshift_estimate(fitted_l1, fitted_l2, std_fitted_l1, std_fitted_l2, line_centre1, line_centre2)

            #best-fit corresponding to this best redshift
            absorber_rest_lam = wavelength / (1 + z_abs_array[k]) # rest-frame conversion of wavelength
            lam_ind = np.where((absorber_rest_lam >= ix0) & (absorber_rest_lam <= ix1))[0]
            lam_fit = absorber_rest_lam[lam_ind]
            nmf_resi = flux[lam_ind]
            error_flux = error[lam_ind]

            fitting_param_for_spectrum[k], fitting_param_std_for_spectrum[k], EW_first_line[k], EW_second_line[k], EW_total[k], fitting_param_pcov_for_spectrum[k] = double_curve_fit(
                index, double_gaussian, lam_fit, nmf_resi, error_fit=error_flux, bounds=bound, init_cond=init_cond, iter_n=1000)

            ## errors on EW
            if not use_covariance:
                EW_first_line_error[k], EW_second_line_error[k], EW_total_error[k] = calculate_ew_errors(fitting_param_for_spectrum[k], fitting_param_std_for_spectrum[k])
            else:
                EW_first_line_error[k], EW_second_line_error[k], EW_total_error[k] = full_covariance_ew_errors(fitting_param_for_spectrum[k], fitting_param_pcov_for_spectrum[k])

            if np.all(np.isnan(EW_first_line[k])) or np.all(np.isnan(EW_second_line[k])) or np.all(np.isnan(EW_total[k])):
                fitting_param_for_spectrum[k] = np.zeros(nparm)
                fitting_param_std_for_spectrum[k] = np.zeros(nparm)
                EW_first_line[k] = 0
                EW_second_line[k] = 0
                EW_total[k] = 0
                EW_first_line_error[k] = 0
                EW_second_line_error[k] = 0
                EW_total_error[k] = 0
                z_abs_err[k] = 0
        else:
            EW_first_line[k] = 0
            EW_second_line[k] = 0
            EW_total[k] = 0
            EW_first_line_error[k] = 0
            EW_second_line_error[k] = 0
            EW_total_error[k] = 0
            fitting_param_for_spectrum[k] = np.zeros(nparm)
            fitting_param_std_for_spectrum[k] = np.zeros(nparm)
            z_abs_err[k] = 0

    return (
        z_abs_array, z_abs_err, fitting_param_for_spectrum, fitting_param_std_for_spectrum,
        EW_first_line, EW_second_line, EW_total,
        EW_first_line_error, EW_second_line_error, EW_total_error
    )
