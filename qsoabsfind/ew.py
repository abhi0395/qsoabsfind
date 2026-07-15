"""
This script contains a function to fit a given absorption profile
with a double gaussian and measure equivalent widths.
"""

import numpy as np
try:
    from numpy import trapezoid as _trapezoid
except ImportError:          # NumPy < 2.0
    from numpy import trapz as _trapezoid
from numba import njit
from scipy.optimize import curve_fit
from .utils import double_gaussian
from .absorberutils import redshift_estimate
from .absorberutils import find_z_from_minimum

# Constants
from .constants import lines, oscillator_parameters, doublet_keys
from . import constants as _constants

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
            ftol=_constants.GAUSS_FIT_FTOL, xtol=_constants.GAUSS_FIT_XTOL
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
    amp1, _, sigma1, amp2, _, sigma2 = popt
    amp1_err, _, sigma1_err, amp2_err, _, sigma2_err = perr

    EW1 = amp1 * np.sqrt(np.pi * 2 * sigma1 ** 2)
    EW2 = amp2 * np.sqrt(np.pi * 2 * sigma2 ** 2)

    # Standard quadrature propagation assuming independent parameters.

    EW1_error = EW1 * np.sqrt((amp1_err / amp1) ** 2 + (sigma1_err / sigma1) ** 2)
    EW2_error = EW2 * np.sqrt((amp2_err / amp2) ** 2 + (sigma2_err / sigma2) ** 2)

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
    amp1, _, sigma1, amp2, _, sigma2 = popt

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

def bootstrap_fitting_and_ew(index, nboot, z, wavelength, flux, error, ix0, ix1, bound, amp_ratio, line1, line2, num_iter, best_params=None, nparm=6):

    """Perform bootstrap resampling to estimate uncertainties in fitting parameters and equivalent widths.

    Conducts bootstrap analysis on spectral data to derive robust estimates of double Gaussian
    fitting parameters and equivalent widths with associated uncertainties for a doublet system.

    Each iteration resamples the fit-window pixels with replacement (true bootstrap), uses a
    reduced iteration budget (GAUSS_FIT_BOOT_ITER_FACTOR * num_iter), and when best_params is
    provided draws initial conditions from a narrow normal distribution around the best-fit values
    (warm start) instead of sampling the full bound range.

    Args:
        index (int): Index or identifier for the current fitting process.
        nboot (int): Number of bootstrap iterations to perform.
        z (float): Redshift of the absorption system.
        wavelength (array-like): Observed Wavelength array of the spectrum.
        flux (array-like): Flux array of the spectrum.
        error (array-like): Flux uncertainty array.
        ix0 (int): Starting index of the spectral region to fit.
        ix1 (int): Ending index of the spectral region to fit.
        bound (tuple or list): Bounds for the fitting parameters.
        amp_ratio (float): Expected amplitude ratio between the two lines in the doublet.
        line1 (float): Rest wavelength of the first line in the doublet.
        line2 (float): Rest wavelength of the second line in the doublet.
        num_iter (int): Maximum number of iterations for each fitting attempt.
        best_params (array-like, optional): Best-fit parameters [amp1, c0, sig1, amp2, c1, sig2]
        nparm (int): Number of fitting parameters (default 6).

    Returns:
        tuple: A tuple containing:
            - fit_params_mean (array): Mean values of fitted parameters across bootstrap samples.
            - fit_param_std (array): Standard deviations of fitted parameters.
            - ew1_mean (float): Mean equivalent width of line 1.
            - ew2_mean (float): Mean equivalent width of line 2.
            - ew_total_mean (float): Mean total equivalent width of the doublet.
            - ew1_std (float): Standard deviation of line 1 equivalent width.
            - ew2_std (float): Standard deviation of line 2 equivalent width.
            - ew_total_std (float): Standard deviation of total equivalent width.
    """

    fit_params = np.zeros((nboot, nparm))
    ew1_array = np.zeros(nboot)
    ew2_array = np.zeros(nboot)
    ew_total_array = np.zeros(nboot)

    absorber_rest_lam = wavelength / (1 + z) # rest-frame conversion of wavelength
    lam_ind = np.where((absorber_rest_lam >= ix0) & (absorber_rest_lam <= ix1))[0]
    lam_fit = absorber_rest_lam[lam_ind]
    nmf_resi = flux[lam_ind]
    error_flux = error[lam_ind]

    min_pixels = _constants.MIN_PIXELS_PER_PARAM * nparm
    if nmf_resi.size < min_pixels or np.all(np.isnan(nmf_resi)):
        nan_p = np.full(nparm, np.nan)
        return nan_p, nan_p, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    amp_first_nmf = max(_constants.GAUSS_AMP_MIN, 1 - np.nanmin(nmf_resi))
    amp_second_nmf = min(_constants.GAUSS_AMP_MAX, amp_ratio * amp_first_nmf)

    # reduced iteration budget per bootstrap fit
    boot_iter = max(50, int(num_iter * _constants.GAUSS_FIT_BOOT_ITER_FACTOR))

    # check whether a valid warm-start is available
    spread = _constants.GAUSS_FIT_BOOT_WARM_SPREAD
    use_warm_start = (best_params is not None and not np.any(np.isnan(best_params)))

    n_pix = len(lam_fit)
    for i in range(nboot):
        # resample fit-window pixels with replacement
        boot_idx = np.sort(np.random.choice(n_pix, size=n_pix, replace=True))
        lam_boot  = lam_fit[boot_idx]
        flux_boot = nmf_resi[boot_idx]
        err_boot  = error_flux[boot_idx]

        # initial conditions drawn near best-fit, else wide random draw
        if use_warm_start:
            if bound is not None:
                amp1   = np.clip(np.random.normal(best_params[0], spread * best_params[0]), bound[0][0], bound[1][0])
                sigma1 = np.clip(np.random.normal(best_params[2], spread * best_params[2]), bound[0][2], bound[1][2])
                amp2   = np.clip(np.random.normal(best_params[3], spread * best_params[3]), bound[0][3], bound[1][3])
                sigma2 = np.clip(np.random.normal(best_params[5], spread * best_params[5]), bound[0][5], bound[1][5])
            else:
                amp1   = np.clip(np.random.normal(best_params[0], spread * best_params[0]), _constants.GAUSS_AMP_MIN,_constants.GAUSS_AMP_MAX)
                sigma1 = np.clip(np.random.normal(best_params[2], spread * best_params[2]), _constants.GAUSS_SIGMA_INIT_MIN, _constants.GAUSS_SIGMA_INIT_MAX)
                amp2   = np.clip(np.random.normal(best_params[3], spread * best_params[3]), _constants.GAUSS_AMP_MIN, _constants.GAUSS_AMP_MAX)
                sigma2 = np.clip(np.random.normal(best_params[5], spread * best_params[5]), _constants.GAUSS_SIGMA_INIT_MIN, _constants.GAUSS_SIGMA_INIT_MAX)
        else:
            amp1, amp2 = amp_first_nmf, amp_second_nmf
            if bound is not None:
                sigma1 = np.random.uniform(bound[0][2], bound[1][2])
                sigma2 = np.random.uniform(bound[0][5], bound[1][5])
            else:
                sigma1 = sigma2 = np.random.uniform(_constants.GAUSS_SIGMA_INIT_MIN, _constants.GAUSS_SIGMA_INIT_MAX)

        init_cond = [amp1, line1, sigma1, amp2, line2, sigma2]
        fit_params[i], _, ew1_array[i], ew2_array[i], ew_total_array[i], _ = double_curve_fit(
            index, double_gaussian, lam_boot, flux_boot, error_fit=err_boot, bounds=bound, init_cond=init_cond, maxefv=boot_iter)

    fit_params_mean = np.nanmean(fit_params, axis=0)
    fit_param_std = np.nanstd(fit_params, axis=0)
    ew1_mean = np.nanmean(ew1_array)
    ew2_mean = np.nanmean(ew2_array)
    ew_total_mean = np.nanmean(ew_total_array)
    ew1_std = np.nanstd(ew1_array)
    ew2_std = np.nanstd(ew2_array)
    ew_total_std = np.nanstd(ew_total_array)

    return fit_params_mean, fit_param_std, ew1_mean, ew2_mean, ew_total_mean, ew1_std, ew2_std, ew_total_std

@njit
def quick_significance_test(flux_norm, fitted_model, error,
                                  fitted_params=None, wavelength_rest=None,
                                  n_pixels=5):
    """
    Significance test computing delta_chi2 independently for each line in the
    doublet.  For each line the chi-squared improvement is evaluated over the
    ``n_pixels`` window centred on that line's fitted position, so the two
    values are completely independent.

    Args:
        flux_norm: Normalized flux array
        fitted_model: Fitted double Gaussian model
        error: Error array
        fitted_params: Fitted parameters [amp1, center1, sigma1, amp2, center2, sigma2]
        wavelength_rest: Wavelength array (same frame as fitted_params)
        n_pixels: Number of pixels around each line center to check

    Returns:
        tuple: (delta_chi2_line1, delta_chi2_line2)
            Per-line delta chi-square values.  A value of 0 is returned for a
            line if its surrounding pixels are not all below the continuum level.
    """
    if flux_norm.size == 0 or error.size == 0:
        return 0.0, 0.0

    dchi2_line1 = 0.0
    dchi2_line2 = 0.0

    if fitted_params is not None and wavelength_rest is not None and wavelength_rest.size > 0:
        # --- Line 1 ---
        lc1 = fitted_params[1]
        idx1 = np.argmin(np.abs(wavelength_rest - lc1))
        s1 = max(0, idx1 - n_pixels)
        e1 = min(len(flux_norm), idx1 + n_pixels + 1)
        pix1 = flux_norm[s1:e1]
        if np.all(pix1 < 1.0):
            mod1 = fitted_model[s1:e1]
            err1 = error[s1:e1]
            cont1 = np.ones(e1 - s1)
            dchi2_line1 = (np.sum(((pix1 - cont1) / err1) ** 2)
                          - np.sum(((pix1 - mod1) / err1) ** 2))

        # --- Line 2 ---
        lc2 = fitted_params[4]
        idx2 = np.argmin(np.abs(wavelength_rest - lc2))
        s2 = max(0, idx2 - n_pixels)
        e2 = min(len(flux_norm), idx2 + n_pixels + 1)
        pix2 = flux_norm[s2:e2]
        if np.all(pix2 < 1.0):
            mod2 = fitted_model[s2:e2]
            err2 = error[s2:e2]
            cont2 = np.ones(e2 - s2)
            dchi2_line2 = (np.sum(((pix2 - cont2) / err2) ** 2)
                          - np.sum(((pix2 - mod2) / err2) ** 2))

    return dchi2_line1, dchi2_line2

def reduced_chi2_doublet_lines(
    lam_obs,
    flux,
    error,
    z_abs,
    params,
    n_sigma_window=3.0,
    min_pixels_each_side=3,
    max_pixels_each_side=15,
    param_count_per_line=3,
):
    """
    Compute local reduced chi2 separately for both lines of a fitted doublet.

    Args:
        lam_obs : array
            Observed-frame wavelength array.
        flux : array
            Continuum-normalized flux array.
        error : array
            Continuum-normalized error array.
        z_abs : float
            Absorber redshift.
        params : array-like
            Gaussian parameters [a1, c1, sigma1, a2, c2, sigma2],
            where c1, sigma1, c2, sigma2 are in rest-frame Angstrom.
        n_sigma_window : float
            Half-width of local chi2 window in units of fitted sigma.
        min_pixels_each_side : int
            Minimum number of pixels on each side of fitted line center.
        max_pixels_each_side : int
            Maximum number of pixels on each side of fitted line center.
        param_count_per_line : int
            Effective number of fitted parameters per line used for ndof.
            For one Gaussian line this is usually 3: amplitude, center, sigma.

    Returns:
        tuple
            (redchi2_line1, redchi2_line2)
    """

    lam_obs = np.asarray(lam_obs, dtype=float)
    flux = np.asarray(flux, dtype=float)
    error = np.asarray(error, dtype=float)
    params = np.asarray(params, dtype=float)

    if (
        lam_obs.size == 0 or flux.size == 0 or error.size == 0 or
        lam_obs.size != flux.size or flux.size != error.size or
        params.size != 6 or not np.isfinite(z_abs)
    ):
        return np.nan, np.nan

    # Work in rest frame because params are rest-frame.
    lam_rest = lam_obs / (1.0 + z_abs)

    a1, c1, sigma1, a2, c2, sigma2 = params

    model = double_gaussian(lam_rest, a1, c1, sigma1, a2, c2, sigma2)

    def get_window(line_center, sigma):
        idx = np.argmin(np.abs(lam_rest - line_center))

        dw = np.nanmedian(np.diff(lam_rest))
        if not np.isfinite(dw) or dw <= 0 or not np.isfinite(sigma) or sigma <= 0:
            return None, None

        half_pix = int(np.ceil(n_sigma_window * sigma / dw))

        half_pix = max(half_pix, min_pixels_each_side)
        half_pix = min(half_pix, max_pixels_each_side)

        s = max(0, idx - half_pix)
        e = min(len(lam_rest), idx + half_pix + 1)

        return s, e

    def one_line_redchi2(line_center, sigma):
        s, e = get_window(line_center, sigma)

        if s is None or e is None or e <= s:
            return np.nan

        pix = flux[s:e]
        mod = model[s:e]
        err = error[s:e]

        good = (
            np.isfinite(pix) &
            np.isfinite(mod) &
            np.isfinite(err) &
            (err > 0)
        )

        n_good = np.count_nonzero(good)

        if n_good <= param_count_per_line:
            return np.nan

        pix = pix[good]
        mod = mod[good]
        err = err[good]

        chi2 = np.sum(((pix - mod) / err) ** 2)
        ndof = n_good - param_count_per_line

        return chi2 / ndof if ndof > 0 else np.nan

    redchi2_line1 = one_line_redchi2(c1, sigma1)
    redchi2_line2 = one_line_redchi2(c2, sigma2)

    return redchi2_line1, redchi2_line2

def _extract_rest_frame_spectrum(wavelength, flux, error, z, ix0, ix1):
    """Slice spectrum into the rest frame defined by redshift *z* between ix0 and ix1."""
    rest_lam = wavelength / (1 + z)
    ind = np.where((rest_lam >= ix0) & (rest_lam <= ix1))[0]
    return rest_lam[ind], flux[ind], error[ind]


def _z_from_rest_fit(params, std, z_k, lc1, lc2):
    """Estimate redshift and error from rest-frame double-Gaussian fit parameters."""
    scale = 1 + z_k
    return redshift_estimate(
        params[1] * scale, params[4] * scale,
        std[1]   * scale, std[4]   * scale,
        lc1, lc2,
    )

def _return_continuum_systematic_error_on_ew(line1, line2, sigma1, sigma2, ew1, ew2, rest_lam, continuum_error_frac=0.05, n_sigma=3):

    w1_lo = line1 - n_sigma * sigma1
    w1_hi = line1 + n_sigma * sigma1
    w2_lo = line2 - n_sigma * sigma2
    w2_hi = line2 + n_sigma * sigma2

    # Clip at midpoint if windows overlap (e.g. close doublets like CIV)
    if w1_hi > w2_lo:
        midpoint = (line1 + line2) / 2.0
        w1_hi = midpoint
        w2_lo = midpoint

    mask1 = (rest_lam >= w1_lo) & (rest_lam <= w1_hi)
    mask2 = (rest_lam >= w2_lo) & (rest_lam <= w2_hi)

    dlam1 = rest_lam[mask1][-1] - rest_lam[mask1][0] if np.any(mask1) else 0.0
    ew1_cont_err = continuum_error_frac * (dlam1  - ew1)
    dlam2 = rest_lam[mask2][-1] - rest_lam[mask2][0] if np.any(mask2) else 0.0
    ew2_cont_err = continuum_error_frac * (dlam2  - ew2)

    return ew1_cont_err, ew2_cont_err


def _fit_single_absorber(index, z_init, wavelength, flux, error,
                         bound, ix0, ix1, line_centre1, line_centre2,
                         amp_ratio, num_iter, window, use_covariance, nboot, nparm):
    """Run the full multi-step fitting pipeline for one absorber system.

    Returns:
        tuple: (z, z_err, params, std, pcov,
                ew1, ew2, ew_total, ew1_err, ew2_err, ew_total_err,
                delta_chi2_line1, delta_chi2_line2)
    """
    zeros_p = np.zeros(nparm)
    zeros_c = np.zeros((nparm, nparm))

    np.random.seed(int(z_init * 1e6) % 2**32)

    # ========== FIRST REDSHIFT REFINEMENT ==========
    z1 = find_z_from_minimum(wavelength, flux, line_centre1, z_init, window=window)
    z2 = find_z_from_minimum(wavelength, flux, line_centre2, z_init, window=window)
    z_k = (line_centre1 * z1 + line_centre2 * z2) / (line_centre1 + line_centre2)

    min_pixels = _constants.MIN_PIXELS_PER_PARAM * nparm  # need enough points to constrain the 6-parameter double Gaussian

    lam_fit, nmf_resi, error_flux = _extract_rest_frame_spectrum(
        wavelength, flux, error, z_k, ix0, ix1)

    if nmf_resi.size < min_pixels or np.all(np.isnan(nmf_resi)):
        # Too few pixels or all NaN: return original redshift, everything else zeroed
        return (z_init, 0.0, zeros_p.copy(), zeros_p.copy(), zeros_c.copy(),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    # ========== INITIAL FIT IN REST FRAME ==========
    amp_first  = max(_constants.GAUSS_AMP_MIN, 1 - np.nanmin(nmf_resi))
    amp_second = min(_constants.GAUSS_AMP_MAX, amp_ratio * amp_first)
    uniform = np.random.uniform
    if bound is not None:
        sigma1 = uniform(bound[0][2], bound[1][2])
        sigma2 = uniform(bound[0][5], bound[1][5])
    else:
        sigma1 = sigma2 = uniform(_constants.GAUSS_SIGMA_INIT_MIN, _constants.GAUSS_SIGMA_INIT_MAX)
    init_cond = [amp_first, line_centre1, sigma1, amp_second, line_centre2, sigma2]

    params, std, ew1, ew2, ew_total, _ = double_curve_fit(
        index, double_gaussian, lam_fit, nmf_resi,
        error_fit=error_flux, bounds=bound, init_cond=init_cond, maxefv=num_iter)

    # ========== FIT IN OBSERVED FRAME FOR REDSHIFT ==========
    obs_init = [amp_first,
                params[1] * (1 + z_k), params[2] * (1 + z_k),
                amp_second,
                params[4] * (1 + z_k), params[5] * (1 + z_k)]
    obs_params, obs_std, _, _, _, _ = double_curve_fit(
        index, double_gaussian, lam_fit * (1 + z_k), nmf_resi,
        error_fit=error_flux, bounds=None, init_cond=obs_init, maxefv=num_iter)

    z_k, z_err = redshift_estimate(
        obs_params[1], obs_params[4],
        obs_std[1],    obs_std[4],
        line_centre1, line_centre2)

    # ========== SECOND REDSHIFT REFINEMENT ==========
    z1 = find_z_from_minimum(wavelength, flux, line_centre1, z_k, window=window)
    z2 = find_z_from_minimum(wavelength, flux, line_centre2, z_k, window=window)
    z_k = 0.5 * (z1 + z2)

    # ========== SECOND FIT WITH REFINED REDSHIFT ==========
    lam_fit, nmf_resi, error_flux = _extract_rest_frame_spectrum(
        wavelength, flux, error, z_k, ix0, ix1)

    params, std, ew1, ew2, ew_total, pcov = double_curve_fit(
        index, double_gaussian, lam_fit, nmf_resi,
        error_fit=error_flux, bounds=bound, init_cond=init_cond, maxefv=num_iter)

    z_k, z_err = _z_from_rest_fit(params, std, z_k, line_centre1, line_centre2)

    # ========== FINAL FIT WITH BEST REDSHIFT ==========
    lam_fit, nmf_resi, error_flux = _extract_rest_frame_spectrum(
        wavelength, flux, error, z_k, ix0, ix1)

    params, std, ew1, ew2, ew_total, pcov = double_curve_fit(
        index, double_gaussian, lam_fit, nmf_resi,
        error_fit=error_flux, bounds=bound, init_cond=init_cond, maxefv=_constants.GAUSS_FIT_FINAL_ITER_FACTOR * num_iter)

    # ========== CALCULATE SIGNIFICANCE ==========
    fitted_model = double_gaussian(lam_fit, *params)
    dchi2_line1, dchi2_line2 = quick_significance_test(nmf_resi, fitted_model, error_flux,
                                    fitted_params=params, wavelength_rest=lam_fit, n_pixels=_constants.SIGNIFICANCE_N_PIXELS)

    # ========== BOOTSTRAPPING OR ERROR CALCULATION ==========
    if nboot is not None and nboot > 0:
        params, std, ew1, ew2, ew_total, ew1_err, ew2_err, ew_total_err = bootstrap_fitting_and_ew(
            index, nboot, z_k, wavelength, flux, error,
            ix0, ix1, bound, amp_ratio, line_centre1, line_centre2, num_iter,
            best_params=params, nparm=nparm)
        fitted_model = double_gaussian(lam_fit, *params)
        dchi2_line1, dchi2_line2 = quick_significance_test(nmf_resi, fitted_model, error_flux,
                                        fitted_params=params, wavelength_rest=lam_fit, n_pixels=_constants.SIGNIFICANCE_N_PIXELS)
    elif use_covariance:
        ew1_err, ew2_err, ew_total_err = full_covariance_ew_errors(params, pcov)
    else:
        ew1_err, ew2_err, ew_total_err = calculate_ew_errors(params, std)

    # ========== CHECK FOR NaN VALUES ==========
    if np.isnan(ew1) or np.isnan(ew2) or np.isnan(ew_total):
        # Keep the refined z but zero out all EW results
        return (z_k, 0.0, zeros_p.copy(), zeros_p.copy(), zeros_c.copy(),
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    return (z_k, z_err, params, std, pcov,
            ew1, ew2, ew_total, ew1_err, ew2_err, ew_total_err, dchi2_line1, dchi2_line2)


def measure_absorber_properties_double_gaussian(
    index, wavelength, flux, error, absorber_redshift, bound, use_kernel, d_pix,
    num_iter=_constants.GAUSS_FIT_NUM_ITER, window=_constants.EW_FIT_WINDOW, use_covariance=False, nboot=None, continuum_error_frac=0.05):
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
        use_kernel (str): Kernel type (MgII, FeII, CIV, NV, OVI, SiIV, AlIII).
        d_pix (float): wavelength pixel for tolerance
        window (int): window size for redshift estimate (default 9)
        use_covariance (bool): if want to use full covariance of scipy curvey_fit for EW error calculation (default is False)
        nboot (int): if provided, will perform bootstrapping fitting (default None)

    Returns:
        tuple: Contains the following elements:
            - z_abs_array (numpy.ndarray): Array of absorber redshifts.
            - z_abs_err (numpy.ndarray): Array of redshift errors.
            - fitting_param_for_spectrum (numpy.ndarray): Array of fitting parameters for double Gaussian.
            - fitting_param_std_for_spectrum (numpy.ndarray): Array of errors for fitting parameters.
            - EW_first_line (numpy.ndarray): Mean equivalent width of the first line.
            - EW_second_line (numpy.ndarray): Mean equivalent width of the second line.
            - EW_total (numpy.ndarray): Mean total equivalent width of both lines.
            - EW_first_line_error (numpy.ndarray): Error in the equivalent width of the first line.
            - EW_second_line_error (numpy.ndarray): Error in the equivalent width of the second line.
            - EW_total_error (numpy.ndarray): Total error in the equivalent width of both lines.
            - delta_chi2_line1 (numpy.ndarray): Per-line delta chi-squared for line 1.
            - delta_chi2_line2 (numpy.ndarray): Per-line delta chi-squared for line 2.
    """
    z_abs_array = np.array(absorber_redshift)
    size_array  = z_abs_array.size

    nparm = 6  # 6 parameter double Gaussian
    fitting_param_for_spectrum     = np.zeros((size_array, nparm))
    fitting_param_std_for_spectrum = np.zeros((size_array, nparm))
    EW_first_line        = np.zeros(size_array, dtype='float32')
    EW_second_line       = np.zeros(size_array, dtype='float32')
    EW_first_line_error  = np.zeros(size_array, dtype='float32')
    EW_second_line_error = np.zeros(size_array, dtype='float32')
    EW_total             = np.zeros(size_array, dtype='float32')
    EW_total_error       = np.zeros(size_array, dtype='float32')
    z_abs_err       = np.zeros(size_array, dtype='float32')
    delta_chi2_line1 = np.zeros(size_array, dtype='float32')
    delta_chi2_line2 = np.zeros(size_array, dtype='float32')

    line_centre1, line_centre2 = return_line_centers(use_kernel)
    amp_ratio = (oscillator_parameters[f'{use_kernel}_f2']
                 / oscillator_parameters[f'{use_kernel}_f1'])

    ix0 = line_centre1 - d_pix * _constants.FIT_WINDOW_HALF_WIDTH
    ix1 = line_centre2 + d_pix * _constants.FIT_WINDOW_HALF_WIDTH

    if size_array == 0:
        return (
            z_abs_array, z_abs_err, fitting_param_for_spectrum, fitting_param_std_for_spectrum,
            EW_first_line, EW_second_line, EW_total,
            EW_first_line_error, EW_second_line_error, EW_total_error,
            delta_chi2_line1, delta_chi2_line2
        )

    for k in range(size_array):
        (z_abs_array[k], z_abs_err[k],
         fitting_param_for_spectrum[k], fitting_param_std_for_spectrum[k], _,
         EW_first_line[k], EW_second_line[k], EW_total[k],
         EW_first_line_error[k], EW_second_line_error[k], EW_total_error[k],
         delta_chi2_line1[k], delta_chi2_line2[k]) = _fit_single_absorber(
            index, absorber_redshift[k], wavelength, flux, error,
            bound, ix0, ix1, line_centre1, line_centre2,
            amp_ratio, num_iter, window, use_covariance, nboot, nparm)

        # continuum error contribution in EW measurements
        ew1_cont_err, ew2_cont_err = _return_continuum_systematic_error_on_ew(
            line_centre1, line_centre2,  fitting_param_for_spectrum[k][2],  fitting_param_for_spectrum[k][5], EW_first_line[k], EW_second_line[k], wavelength/(1+z_abs_array[k]), continuum_error_frac=continuum_error_frac, n_sigma=_constants.SNR_NSIG)

        EW_first_line_error[k] = np.sqrt(EW_first_line_error[k]**2 + ew1_cont_err**2)
        EW_second_line_error[k] = np.sqrt(EW_second_line_error[k]**2 + ew2_cont_err**2)
        EW_total_error[k] = np.sqrt(EW_first_line_error[k]**2 + EW_second_line_error[k]**2)

    return (
        z_abs_array, z_abs_err, fitting_param_for_spectrum, fitting_param_std_for_spectrum,
        EW_first_line, EW_second_line, EW_total,
        EW_first_line_error, EW_second_line_error, EW_total_error,
        delta_chi2_line1, delta_chi2_line2
    )


def trapezoidal_ew(wavelength, residual, error, z, line1, line2, sigma1, sigma2, n_sigma=3, continuum_error_frac=0.05):
    """
    Measure the rest-frame equivalent width (EW) and its 1-sigma uncertainty
    for two absorption lines using the trapezoidal integration method.

    The integration window for each line is normally
    ``[line_centre - n_sigma * sigma, line_centre + n_sigma * sigma]``
    evaluated in the rest frame.  If the two windows overlap (as can occur
    for close doublets such as CIV), each window is clipped at the midpoint
    between the two line centres to prevent double-counting.  Per-pixel
    errors are propagated analytically through the trapezoidal-rule weights.

    Args:
        wavelength (numpy.ndarray): Observed wavelength array (Ang).
        residual (numpy.ndarray): Normalised flux array.  Values should be
            close to 1 in the continuum and dip below 1 in absorption.
        error (numpy.ndarray): Per-pixel 1-sigma flux error array.
        z (float): Absorber redshift used to convert to the rest frame.
        line1 (float): Rest-frame wavelength of the first line (Ang).
        line2 (float): Rest-frame wavelength of the second line (Ang).
        sigma1 (float): Gaussian width (1-sigma) of the first line (Ang, rest
            frame) used to define the integration window.
        sigma2 (float): Gaussian width (1-sigma) of the second line (Ang, rest
            frame) used to define the integration window.
        n_sigma (float): Half-width of each integration window expressed in
            units of the corresponding sigma.  Default is 3.

    Returns:
        tuple: ``(ew1, ew2, ew_total, ew1_err, ew2_err, ew_total_err)``

            - *ew1*, *ew2* - rest-frame EW of line 1 and line 2 (Ang).
            - *ew_total* - sum of the two EWs (Ang).
            - *ew1_err*, *ew2_err*, *ew_total_err* - corresponding 1-sigma
              uncertainties (Ang).

            A value of ``NaN`` is returned for any quantity whose integration
            window contains fewer than two pixels.
    """
    rest_lam = wavelength / (1.0 + z)

    def _trapz_ew_and_err(lam, flux, err):
        """Integrate (1 - flux) over *lam* and propagate *err* via trapezoid weights."""
        if lam.size < 2:
            return np.nan, np.nan

        absorption = 1.0 - flux
        ew = _trapezoid(absorption, lam)

        # Trapezoidal-rule weights: each pixel's contribution to the integral
        dlam = np.diff(lam)
        weights = np.empty(lam.size)
        weights[0] = dlam[0] / 2.0
        weights[-1] = dlam[-1] / 2.0
        weights[1:-1] = (dlam[:-1] + dlam[1:]) / 2.0

        ew_err = np.sqrt(np.sum((weights * err) ** 2))
        return ew, ew_err

    w1_lo = line1 - n_sigma * sigma1
    w1_hi = line1 + n_sigma * sigma1
    w2_lo = line2 - n_sigma * sigma2
    w2_hi = line2 + n_sigma * sigma2

    # Clip at midpoint if windows overlap (e.g. close doublets like CIV)
    if w1_hi > w2_lo:
        midpoint = (line1 + line2) / 2.0
        w1_hi = midpoint
        w2_lo = midpoint

    mask1 = (rest_lam >= w1_lo) & (rest_lam <= w1_hi)
    mask2 = (rest_lam >= w2_lo) & (rest_lam <= w2_hi)

    ew1, ew1_err = _trapz_ew_and_err(rest_lam[mask1], residual[mask1], error[mask1])
    ew2, ew2_err = _trapz_ew_and_err(rest_lam[mask2], residual[mask2], error[mask2])

    dlam1 = rest_lam[mask1][-1] - rest_lam[mask1][0] if np.any(mask1) else 0.0
    ew1_cont_err = continuum_error_frac * (dlam1  - ew1)
    dlam2 = rest_lam[mask2][-1] - rest_lam[mask2][0] if np.any(mask2) else 0.0
    ew2_cont_err = continuum_error_frac * (dlam2  - ew2)

    ew1_err = np.sqrt(ew1_err**2 + ew1_cont_err**2)
    ew2_err = np.sqrt(ew2_err**2 + ew2_cont_err**2)

    both_nan = np.isnan(ew1) and np.isnan(ew2)
    if both_nan:
        ew_total, ew_total_err = np.nan, np.nan
    else:
        ew_total = np.nansum([ew1, ew2])
        ew_total_err = np.sqrt(np.nansum([
            0.0 if np.isnan(ew1_err) else ew1_err ** 2,
            0.0 if np.isnan(ew2_err) else ew2_err ** 2,
        ]))

    return {'ew1': ew1, 'ew2': ew2, 'ew_total': ew_total, 'ew1_err': ew1_err, 'ew2_err': ew2_err, 'ew_total_err': ew_total_err}
