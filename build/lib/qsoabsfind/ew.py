import numpy as np
from scipy.optimize import curve_fit
from .constants import lines

# Example usage within double_curve_fit
def double_curve_fit(index, fun_to_run, lam_fit_range, nmf_resi_fit, error_fit, bounds, init_cond, iter_n):
    """
    Fits a double Gaussian function to the provided data.

    Parameters:
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
        - EW1_error (float): Error in the equivalent width of the first Gaussian.
        - EW2_error (float): Error in the equivalent width of the second Gaussian.
        - EW_total_error (float): Total error in the equivalent width of both Gaussians.
    """
    nparm = len(init_cond)
    save_param_array = np.zeros(nparm)
    save_param_error = np.zeros(nparm)
    EW_first = np.nan
    EW_second = np.nan
    EW_total = np.nan

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

    except (RuntimeError, ValueError, TypeError) as e:
        if isinstance(e, TypeError):
            print(f'NMF resi fit size = {nmf_resi_fit.size}')
            print(f'\nIn Main Gaussian script: In double Gaussian fitting: Spec Index = {index} has issues, Check this please\n')
        save_param_array[:] = np.nan
        save_param_error[:] = np.nan

    return save_param_array, save_param_error, EW_first, EW_second, EW_total


def calculate_ew_errors(popt, perr):
    """
    Calculate the errors in the equivalent widths (EW) using the errors in the optimized parameters.

    Parameters:
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

    EW1_error = EW1 * np.sqrt((amp1_err / amp1) ** 2 + (2 * sigma1_err / sigma1) ** 2)
    EW2_error = EW2 * np.sqrt((amp2_err / amp2) ** 2 + (2 * sigma2_err / sigma2) ** 2)

    EW_total_error = np.sqrt(EW1_error ** 2 + EW2_error ** 2)

    return EW1_error, EW2_error, EW_total_error

def full_covariance_ew_errors(popt, pcov):
    """
    With full covariance matrix
    Calculate the errors in the equivalent widths (EW) using the errors in the optimized parameters.

    Parameters:
    popt (numpy.ndarray): Optimized parameters from the curve fitting.
    pcov (numpy.ndarray): Covariance matrix from the curve fitting.

    Returns:
    tuple: Contains the following elements:
        - EW1_error (float): Error in the equivalent width of the first Gaussian.
        - EW2_error (float): Error in the equivalent width of the second Gaussian.
        - EW_total_error (float): Total error in the equivalent width of both Gaussians.
    """
    # Partial derivatives of EW with respect to each parameter
    def partial_derivatives(popt):
        amp1, mean1, sigma1, amp2, mean2, sigma2 = popt
        dEW1_damp1 = np.sqrt(np.pi * 2 * sigma1 ** 2)
        dEW1_dsigma1 = amp1 * np.sqrt(np.pi * 2) * sigma1

        dEW2_damp2 = np.sqrt(np.pi * 2 * sigma2 ** 2)
        dEW2_dsigma2 = amp2 * np.sqrt(np.pi * 2) * sigma2

        return [dEW1_damp1, 0, dEW1_dsigma1, 0, 0, 0], [0, 0, 0, dEW2_damp2, 0, dEW2_dsigma2]

    # Calculate the partial derivatives
    partials_EW1, partials_EW2 = partial_derivatives(popt)

    # Calculate the variance of EW1 and EW2 using the covariance matrix
    EW1_var = np.dot(np.dot(partials_EW1, pcov), partials_EW1)
    EW2_var = np.dot(np.dot(partials_EW2, pcov), partials_EW2)

    # Calculate the errors in EW1 and EW2
    EW1_error = np.sqrt(EW1_var)
    EW2_error = np.sqrt(EW2_var)

    # Total EW error (assuming independence)
    EW_total_error = np.sqrt(EW1_var + EW2_var)

    return EW1_error, EW2_error, EW_total_error

def measure_absorber_properties_double_gaussian(index, wavelength, flux, error, absorber_redshift, bound, use_kernel='Mg', d_pix=0.6, use_covariance=False):
    """
    Measures the properties of each potential absorber by fitting a double Gaussian to the absorption feature
    and measuring the equivalent width (EW) and errors of absorption lines.

    Parameters:
    index (int): Index of the spectrum being fitted.
    wavelength (numpy.ndarray): Array containing common rest frame quasar wavelength.
    flux (numpy.ndarray): Matrix containing the residual flux.
    error (numpy.ndarray): Error array corresponding to the flux.
    absorber_redshift (list): List of potential absorbers identified previously.
    bound (tuple): Bounds for the fitting parameters.
    use_kernel (str, optional): Kernel type ('MgII, FeII, CIV). Default is 'MgII'.
    d_pix (float, optional): wavelength pixel for tolerance (default 0.6A)
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

    np.random.seed(1234) # for reproducibility of initital condition
    z_abs_array = np.array(absorber_redshift)
    size_array = z_abs_array.size

    nparm = 6 # 6 parameter double Gaussian
    fitting_param_for_spectrum = np.zeros((size_array, nparm))
    fitting_param_std_for_spectrum = np.zeros((size_array, nparm))
    EW_first_line = np.zeros(size_array, dtype='float32')
    EW_second_line = np.zeros(size_array, dtype='float32')
    EW_first_line_error = np.zeros(size_array, dtype='float32')
    EW_second_line_error = np.zeros(size_array, dtype='float32')
    EW_total = np.zeros(size_array, dtype='float32')
    EW_total_error = np.zeros(size_array, dtype='float32')

    lines = line_data()

    if use_kernel == 'MgII':
        line_centre1 = lines['MgII_2796']
        line_centre2 = lines['MgII_2803']
    elif use_kernel == 'FeII':
        line_centre1 = lines['FeII_2586']
        line_centre2 = lines['FeII_2600']
    elif use_kernel == 'CIV':
        line_centre1 = lines['CIV_1548']
        line_centre2 = lines['CIV_1550']
    else:
        raise ValueError("Unsupported kernel type. Use 'Mg', 'Fe', or 'CIV'.")

    #defining wwavelength range for Gaussian fitting
    sigma = d_pix*5 # assuming maximum line width of d_pix * 5, can be larger/smaller, but this is a reasonable assumption.
    ix0 = line_centre1 - 5 * sigma
    ix1 = line_centre2 + 5 * sigma

    if size_array == 0:
        return (
            z_abs_array, fitting_param_for_spectrum, fitting_param_std_for_spectrum,
            EW_first_line, EW_second_line, EW_total,
            EW_first_line_error, EW_second_line_error, EW_total_error
        )

    for k in range(size_array):
        absorber_rest_lam = wavelength / (1 + absorber_redshift[k]) # rest-frame conversion of wavelength
        lam_ind = np.where((absorber_rest_lam >= ix0) & (absorber_rest_lam <= ix1))[0]
        lam_fit = absorber_rest_lam[lam_ind]
        nmf_resi = flux[lam_ind]
        error_flux = error[lam_ind]

        if nmf_resi.size > 0 and not np.all(np.isnan(nmf_resi)):
            #random initial condition
            amp_first_nmf = np.nanmin(nmf_resi) + np.random.normal(0, d_pix/10)
            line_first = line_centre1 + np.random.normal(0, d_pix)
            sigma1 = 1.27 + np.random.normal(0, d_pix/2)
            sigma2 = 1.27 + np.random.normal(0, d_pix/2)
            line_second = line_centre2 + np.random.normal(0, d_pix)
            init_cond = [amp_first_nmf, line_first, sigma1, 0.54 * amp_first_nmf, line_second, sigma2]

            fitting_param_for_spectrum[k], fitting_param_std_for_spectrum[k], EW_first_line[k], EW_second_line[k], EW_total[k] = double_curve_fit(
                index, gauss, lam_fit, nmf_resi, error_fit=error_flux, bounds=bound, init_cond=init_cond, iter_n=1000)
            #errors on EW
            if not use_covariance:
                EW_first_line_error[k], EW_second_line_error[k], EW_total_error[k] = calculate_ew_errors(fitting_param_for_spectrum[k], fitting_param_std_for_spectrum[k])
            else:
                EW_first_line_error[k], EW_second_line_error[k], EW_total_error[k] = full_covariance_ew_errors(fitting_param_for_spectrum[k], fitting_param_std_for_spectrum[k])

            if np.all(np.isnan(EW_first_line[k])) or np.all(np.isnan(EW_second_line[k])) or np.all(np.isnan(EW_total[k])):
                fitting_param_for_spectrum[k] = np.zeros(nparm)
                fitting_param_std_for_spectrum[k] = np.zeros(nparm)
                EW_first_line[k] = 0
                EW_second_line[k] = 0
                EW_total[k] = 0
                EW_first_line_error[k] = 0
                EW_second_line_error[k] = 0
                EW_total_error[k] = 0
        else:
            EW_first_line[k] = 0
            EW_second_line[k] = 0
            EW_total[k] = 0
            EW_first_line_error[k] = 0
            EW_second_line_error[k] = 0
            EW_total_error[k] = 0
            fitting_param_for_spectrum[k] = np.zeros(nparm)
            fitting_param_std_for_spectrum[k] = np.zeros(nparm)

    return (
        z_abs_array, fitting_param_for_spectrum, fitting_param_std_for_spectrum,
        EW_first_line, EW_second_line, EW_total,
        EW_first_line_error, EW_second_line_error, EW_total_error
    )
