"""
This script contains a utility functions that are repeatedly
called during the absorber search in QSO spectra.
"""

import numpy as np
from numba import jit
import time
import logging
from astropy.table import Table
from scipy.stats import chi2


# Constants -- imported via the module object so that startup-time patches
# (applied in parallel_convolution.main) propagate here automatically.
from .constants import lines, speed_of_light, doublet_keys
from . import constants as _constants

logger = logging.getLogger(__name__)


@jit(nopython=True)
def find_valid_indices(our_z, residual_our_z, lam_search, conv_arr, sigma_cr, coeff_sigma, beta, line1, line2, logwave):
    """
    Find valid indices based on thresholding in the convolved array.

    Args:
        our_z (array): Array of redshift values.
        residual_our_z (array): Array of residual values at the redshift positions.
        lam_search (array): Array of wavelengths.
        conv_arr (array): Convolved array.
        sigma_cr (array): Local sigma values.
        coeff_sigma (float): Coefficient for sigma.
        beta (float): Oscillator strength ratio.
        line1 (float): First line wavelength.
        line2 (float): Second line wavelength.
        logwave (bool): if wavelength pixels are on fixed log-scale (e.g. SDSS)

    Returns:
        Tuple: Arrays of new redshift values and new residual values.
    """
    new_our_z = []
    new_res_arr = []
    npix = _constants.CANDIDATE_VALIDATION_NPIX  # number of pixels around a line minima
    del_lam = line2 - line1
    line_centre = 0.5 * (line1 + line2)

    for k in range(1, len(our_z)):
        z_plus_one = (1 + our_z[k])
        lam_check = (line_centre - del_lam) * z_plus_one
        lam_check1 = (line_centre + del_lam) * z_plus_one
        if logwave:
            lam_check_thresh1 = np.abs(lam_check * (10**(npix * 0.0001) - 1))
            lam_check_thresh2 = np.abs(lam_check1 * (10**(npix * 0.0001) - 1))
        else:
            lam_check_thresh1= npix * (lam_search[1]-lam_search[0])
            lam_check_thresh2 = lam_check_thresh1

        ind_check = (lam_search >= lam_check - lam_check_thresh1) & (lam_search <= lam_check + lam_check_thresh1)
        ind_check1 = (lam_search >= lam_check1 - lam_check_thresh2) & (lam_search <= lam_check1 + lam_check_thresh2)

        if not (np.all(np.isnan(conv_arr[ind_check])) and np.all(np.isnan(conv_arr[ind_check1]))):
            conv_arr1 = conv_arr[ind_check]
            conv_arr2 = conv_arr[ind_check1]

            sec_thr1 = np.nanmedian(conv_arr) - coeff_sigma / beta * sigma_cr[ind_check]
            sec_thr2 = np.nanmedian(conv_arr) - coeff_sigma / beta * sigma_cr[ind_check1]

            if np.all(conv_arr1 <= sec_thr1) or np.all(conv_arr2 <= sec_thr2):
                new_our_z.append(our_z[k])
                new_res_arr.append(residual_our_z[k])

    return new_our_z, new_res_arr

@jit(nopython=True)
def estimate_local_sigma_conv_array(conv_array, pm_pixel):
    """
    Estimate the local standard deviation for each element in the
    convolution array over a window defined by pm_pixel.

    Args:
        conv_array (numpy.ndarray): Input convolution array for which local standard deviations are to be calculated.
        pm_pixel (int): Number of pixels defining the window size around each element for local standard deviation calculation.

    Returns:
        numpy.ndarray: Array of local standard deviations.
    """
    nsize = conv_array.size
    pivot = pm_pixel // 2
    sigma_cr = np.zeros(nsize, dtype=np.float32)

    for i in range(nsize):
        start = max(0, i - pivot)
        end = min(nsize, i + pivot + 1)
        sigma_cr[i] = np.nanstd(conv_array[start:end])

    return sigma_cr

@jit(nopython=True)
def calculate_doublet_ratio(ew1, ew2, ew1_error, ew2_error, f1, f2):
    """
    Calculate the doublet ratio and its associated error.

    Args:
        ew1 (float): Equivalent width of the first line.
        ew2 (float): Equivalent width of the second line.
        ew1_error (float): Error associated with the first equivalent width.
        ew2_error (float): Error associated with the second equivalent width.
        f1 (float): oscillator strength of first line
        f2 (float): oscillator strength of second line

    Returns:
        tuple: A tuple containing:
            - doublet_ratio (float): The ratio of ew1 to ew2.
            - doublet_ratio_error (float): The propagated error for the doublet ratio.
    """
    # Calculate the doublet ratio
    doublet_ratio = ew1 / ew2 # default definition
    if f1 < f2:
        doublet_ratio = ew2 / ew1

    # Propagate the error using standard error propagation formula for division
    doublet_ratio_error = doublet_ratio * np.sqrt((ew1_error / ew1) ** 2 + (ew2_error / ew2) ** 2)

    return doublet_ratio, doublet_ratio_error


@jit(nopython=True)
def estimate_snr_for_lines(l1, l2, sig1, sig2, lam_rest, residual, error, log):
    """
    Estimate S/N of the doublet lines.

    Args:
        l1 (float): First wavelength to check around.
        l2 (float): Second wavelength to check around.
        sig1 (float): fitted width of the first line
        sig2 (float): fitted width of second line
        lam_rest (numpy.ndarray): Rest-frame wavelengths.
        residual (numpy.ndarray): Residual flux values.
        error (numpy.ndarray): Error values corresponding to the residuals.
        log (bool): if wavelength bins are on log scale

    Returns:
        tuple: Integrated signal-to-noise ratios (SNR) around the specified wavelengths.
               Returns (mean_sn1, mean_sn2).
    """
    if sig1 is None or sig2 is None:
        dpix = _constants.SNR_DEFAULT_DPIX
        if log:
            delta1 = np.abs(l1 * (10**(dpix * 0.0001) - 1))
            delta2 = np.abs(l2 * (10**(dpix * 0.0001) - 1))
        else:
            delta1 = dpix * (lam_rest[1]-lam_rest[0])
            delta2 = delta1
    else:
        nsig = _constants.SNR_NSIG  # for gaussian sigma to account how far we want to go.
        delta1, delta2 = nsig * sig1, nsig * sig2

    ind1 = np.where((lam_rest > l1 - delta1) & (lam_rest < l1 + delta1))[0]
    ind2 = np.where((lam_rest > l2 - delta2) & (lam_rest < l2 + delta2))[0]

    resi1 = residual[ind1]
    resi2 = residual[ind2]

    err1 = error[ind1]
    err2 = error[ind2]

    median = 1  # Assuming median residual value is 1, as it continuum-normalized

    diff1 = median - resi1
    diff2 = median - resi2

    sum_diff1 = np.nansum(diff1)
    sum_diff2 = np.nansum(diff2)
    sum_err1 = np.sqrt(np.nansum(err1**2))
    sum_err2 = np.sqrt(np.nansum(err2**2))

    mean_sn1, mean_sn2 = -1, -1 # in case failure

    if sum_err1 != 0 and sum_err2 !=0 and np.isfinite(sum_err1) and np.isfinite(sum_err2):
        mean_sn1 = sum_diff1 / sum_err1
        mean_sn2 = sum_diff2 / sum_err2

    return mean_sn1, mean_sn2


def check_local_continuum_return(
    wave, flux, error, z_abs,
    lam1_rest,
    lam2_rest,
    sigma_rest1,
    sigma_rest2,
    n_sigma_inner=3.0,
    n_sigma_side=4.0,
    min_pixels=5,
    min_median_flux=0.9,
):
    """
    Simple veto for broad troughs/blends.

    Returns True if the residual returns close to continuum
    on both sides of the Na I doublet.
    """

    lam1 = lam1_rest * (1.0 + z_abs)
    lam2 = lam2_rest * (1.0 + z_abs)
    sigma_obs1 = sigma_rest1 * (1.0 + z_abs)
    sigma_obs2 = sigma_rest2 * (1.0 + z_abs)

    left_edge  = lam1 - n_sigma_inner * sigma_obs1
    right_edge = lam2 + n_sigma_inner * sigma_obs2

    side_width1 = n_sigma_side * sigma_obs1
    side_width2 = n_sigma_side * sigma_obs2

    left = (wave > left_edge - side_width1) & (wave < left_edge)
    right = (wave > right_edge) & (wave < right_edge + side_width2)

    def side_ok(mask):
        good = mask & np.isfinite(flux) & np.isfinite(error) & (error > 0)
        if np.count_nonzero(good) < min_pixels:
            return False

        med_flux = np.nanmedian(flux[good])
        med_err = np.nanmedian(error[good]) / np.sqrt(np.count_nonzero(good))

        return med_flux >= (min_median_flux - med_err)

    return bool(side_ok(left) and side_ok(right))


def group_contiguous_pixel(data, resi, avg):
    """
    Arrange data into groups where successive elements differ by less than the average difference.

    Args:
        data (np.ndarray): List of absorbers for each spectrum.
        resi (np.ndarray): Corresponding residual.
        avg (float): Maximum allowed difference to group data.

    Returns:
        tuple: Final list with contiguous pixels grouped, and corresponding residuals if provided.
    """
    # Check if resi is provided as an array
    if resi is not None:
        has_residuals = True
    else:
        has_residuals = False

    # Sort the input data
    ind_srt = np.argsort(data)
    sorted_data = data[ind_srt]

    if has_residuals:
        sorted_resi = resi[ind_srt]

    groups = []
    resi_groups = []

    groups.append([sorted_data[0]])
    if has_residuals:
        resi_groups.append([sorted_resi[0]])

    for i in range(1, len(sorted_data)):
        if sorted_data[i] - groups[-1][-1] < avg:
            groups[-1].append(sorted_data[i])
            if has_residuals:
                resi_groups[-1].append(sorted_resi[i])
        else:
            groups.append([sorted_data[i]])
            if has_residuals:
                resi_groups.append([sorted_resi[i]])

    if has_residuals:
        return groups, resi_groups
    else:
        return groups

def group_and_select_weighted_redshift(redshifts, fluxes, residual, lam_obs, line1, line2, delta_z):
    """
    Group contiguous redshifts and select the highest weighted
    (corresponding to minimum flux) redshift from each group.

    Args:
        redshifts (list or numpy.ndarray): List of candidate absorber redshifts.
        fluxes (list or numpy.ndarray): Corresponding residual fluxes near each redshift.
        residual (numpy.ndarray): Full residual flux array aligned with ``lam_obs``.
        lam_obs (numpy.ndarray): Observed-frame wavelength array.
        line1 (float): Rest-frame wavelength of the first doublet line (Angstrom).
        line2 (float): Rest-frame wavelength of the second doublet line (Angstrom).
        delta_z (float): Maximum redshift difference to consider two candidates contiguous.

    Returns:
        list: Best redshift from each contiguous group (minimum-flux weighted selection).
    """

    # Ensure inputs are numpy arrays for easy manipulation
    all_redshifts = np.array(redshifts)
    fluxes = np.array(fluxes)
    redshifts = []
    for z in all_redshifts:
        z1 = find_z_from_minimum(lam_obs, residual, line1, z, window=_constants.REDSHIFT_REFINE_WINDOW)
        z2 = find_z_from_minimum(lam_obs, residual, line2, z, window=_constants.REDSHIFT_REFINE_WINDOW)
        new_z = (line1 * z1 + line2 * z2) / (line1 + line2)
        redshifts.append(new_z)

    redshifts = np.array(redshifts)

    # Check if redshifts array is empty
    if len(redshifts) == 0:
        return []

    # Initialize lists to store results
    best_redshifts = []

    # Sort redshifts and corresponding data
    sorted_indices = np.argsort(redshifts)
    redshifts = redshifts[sorted_indices]
    fluxes = fluxes[sorted_indices]

    # Initialize the first group
    current_group = [sorted_indices[0]]

    # Group contiguous redshifts
    for i in range(1, len(redshifts)):
        if redshifts[i] - redshifts[i - 1] <= delta_z:
            current_group.append(sorted_indices[i])
        else:
            # Select the redshift with the highest weight (i.e. minimum flux) in the current group
            best_index = min(current_group, key=lambda idx: fluxes[idx])
            best_redshifts.append(redshifts[best_index])
            # Start a new group
            current_group = [sorted_indices[i]]

    # Select the best redshift in the last group
    if current_group:
        best_index = min(current_group, key=lambda idx: fluxes[idx])
        best_redshifts.append(redshifts[best_index])

    return best_redshifts

def find_z_from_minimum(wavelength, residual, line_rest, z_guess, window=_constants.REDSHIFT_REFINE_WINDOW, log=False):
    """Estimate absorber redshift from the minimum flux near the expected line center.

    Given an initial redshift guess, this function identifies a symmetric window
    around the expected observed-frame line center and finds the pixel with the
    minimum residual (flux) within that window. The wavelength of that minimum is
    converted back to a refined redshift for the line.

    Args:
        wavelength (numpy.ndarray): Observed-frame wavelength array, monotonically
            increasing and aligned with `residual`.
        residual (numpy.ndarray): Residual/flux array aligned with `wavelength`.
            May contain NaNs; the minimum is computed with `np.nanargmin`.
        line_rest (float): Rest-frame wavelength (in the same units as `wavelength`)
            of the spectral line used for refinement (e.g., 1548.204 A for C IV).
        z_guess (float): Initial absorber redshift guess.
        window (int, optional): Half-window size in **pixels** for the local search
            around the expected line center. Defaults to 9. The search range is
            +/- `window` x (wavelength pixel spacing).
        log (bool): if wavelenght ins log-scale (default False)

    Returns:
        float: Refined redshift estimate computed as `(lambda_min / line_rest) - 1`, where
        `lambda_min` is the observed-frame wavelength at the minimum residual within the
        search window. If no pixels fall within the window, returns `z_guess`.

    Note:
        - If the search window contains only NaNs, `np.nanargmin` will raise a
          `ValueError`. Consider pre-filtering `residual` or guarding with
          `np.isfinite` if this is a possibility in your data.
        - The window is defined in **observed-frame** wavelength by converting the
          pixel count to delta lambda using the local pixel spacing.
    """
    lam_expected = line_rest * (1 + z_guess)
    if log:
        del_log_lam = np.log10(wavelength[1]) - np.log10(wavelength[0])
        delta = lam_expected * (10**(window * del_log_lam) - 1)
    else:
        delta_lam = (wavelength[1] - wavelength[0])
        delta = window * delta_lam
    mask = (wavelength > lam_expected - delta) & (wavelength < lam_expected + delta)

    if np.any(mask):
        idx_min = np.nanargmin(residual[mask])
        lam_min = wavelength[mask][idx_min]
        return lam_min / line_rest - 1
    else:
        return z_guess  # fallback

def median_selection_after_combining(combined_final_our_z, lam_search, residual, d_pix, use_kernel, delta_z, window=_constants.REDSHIFT_REFINE_WINDOW, gamma=_constants.MEDIAN_WEIGHT_GAMMA):
    """
    Perform grouping and weighted mean from the list of all potentially
    identified absorbers after combining from all the runs with different
    kernel widths.

    Args:
        combined_final_our_z (list): List of potential absorbers identified for each spectrum.
        lam_search (numpy.ndarray): Wavelength search array.
        residual (numpy.ndarray): Residual values corresponding to the absorbers.
        d_pix (float): Pixel separation tolerance in wavelength (default 0.6 Angstrom).
        use_kernel (str): Kernel/absorber type (e.g. MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, NaI).
        delta_z (float): Maximum redshift difference to consider two candidates contiguous.
        window (int): window size for redshift estimate (default 9)
        gamma (int): power for lambda to use in 1/lam**gamma weighting scheme (default 4)

    Returns:
        list: List after grouping contiguous pixels for each spectrum.
    """

    thresh  = lines[doublet_keys[use_kernel][0]]
    thresh1  = lines[doublet_keys[use_kernel][1]]
    new_z = []
    for z in combined_final_our_z:
        z1 = find_z_from_minimum(lam_search, residual, thresh, z, window=window)
        z2 = find_z_from_minimum(lam_search, residual, thresh1, z, window=window)
        nz = (thresh * z1 + thresh1 * z2) / (thresh + thresh1)
        new_z.append(nz)

    combined_final_our_z = new_z

    z_ind = []  # Final list of median redshifts for each spectrum
    ct = _constants.CANDIDATE_DEDUP_CT
    if len(combined_final_our_z) > 1:
        abs_list = np.array(combined_final_our_z)

        # Grouping contiguous pixels separated by delta_z on redshift scale
        z_temp = group_contiguous_pixel(abs_list, resi=None, avg=delta_z)

        for group in z_temp:
            temp_z = []
            res_z = []
            for zabs in group:
                lam_abs = lam_search / (1 + zabs)
                ind_z = np.where((lam_abs >= thresh - ct * d_pix) & (lam_abs <= thresh + ct * d_pix))[0]

                if not np.all(np.isnan(residual[ind_z])):
                    res_ind_z = np.nanmin(residual[ind_z])
                    if res_ind_z != 0:
                        temp_z.append(zabs / res_ind_z**gamma)
                        res_z.append(1 / res_ind_z**gamma)

            if temp_z and res_z:
                abs_z = np.nansum(temp_z) / np.nansum(res_z)
                z_ind.append(abs_z)

        return z_ind
    else:
        return combined_final_our_z

def check_absorber_selection(qso_id, zabs, gaussian_parameters, bound,
                             lower_del_lam, c0, c1, upper_del_lam,
                             sn1, sn_line1, sn2, sn_line2,
                             vel1, vel2, min_dr, dr, max_dr,
                             ew1_snr, ew2_snr, delta_chi2_line1, delta_chi2_line2,
                             fit_param_std=None,
                             conf_level=0.95, vmax=_constants.MAX_VEL_DISPERSION, verbose=False):
    """Check absorber selection criteria, print details, and count satisfied conditions.

    Evaluates whether a candidate absorber passes various selection criteria based on
    spectral line properties, signal-to-noise ratios, velocity constraints, and
    statistical significance.

    Args:
        qso_id (str): Unique identifier for the quasar/QSO.
        zabs (float): Absorption redshift of the candidate system.
        gaussian_parameters (array-like): Fitted parameters for the double Gaussian model,
            typically [amp1, center1, width1, amp2, center2, width2].
        bound (bool): Bounds for Gaussian fit parameters.
        lower_del_lam (float): Lower wavelength offset/deviation.
        c0 (float): Central wavelength or reference wavelength for line 1.
        c1 (float): Central wavelength or reference wavelength for line 2.
        upper_del_lam (float): Upper wavelength offset/deviation.
        sn1 (float): Signal-to-noise ratio near line 1.
        sn_line1 (float): Signal-to-noise ratio thresholde at line 1 center.
        sn2 (float): Signal-to-noise ratio for continuum near line 2.
        sn_line2 (float): Signal-to-noise ratio thresholde at line 2 center.
        vel1 (float): Velocity width of line 1 component in km/s.
        vel2 (float): Velocity width of line 2 component in km/s.
        min_dr (float): Minimum doublet ratio threshold.
        dr (float): Measured doublet ratio (e.g., CIV 1548/1550 ratio).
        max_dr (float): Maximum doublet ratio threshold.
        ew1_snr (float): Equivalent width signal-to-noise ratio for line 1.
        ew2_snr (float): Equivalent width signal-to-noise ratio for line 2.
        delta_chi2_line1 (float): Per-line delta chi-squared for line 1.
        delta_chi2_line2 (float): Per-line delta chi-squared for line 2.
        fit_param_std (array-like, optional): Standard errors of the 6 Gaussian fit
            parameters [amp1_err, c0_err, sig1_err, amp2_err, c1_err, sig2_err]. Defaults to None (check skipped).
        conf_level (float, optional): Confidence level for statistical significance.
            Defaults to 0.95 (95% confidence).
        vmax (float, optional): Maximum allowed velocity difference between components
            in km/s. Defaults to 120.

    Returns:
        bool: True if the absorber passes the selection criteria, False otherwise.

    Note:
        The function evaluates multiple selection criteria including:
        - Wavelength bounds for both lines
        - S/N thresholds for continuum and line centers
        - Velocity constraints between components
        - Doublet ratio physical limits
        - Equivalent width significance
        - Statistical significance via delta chi-squared test
        - Fit-parameter SNR: abs(param) / param_err > FIT_PARAM_SNR (when fit_param_std supplied)

        Prints detailed information about each criterion and whether it passes.

    """

    critical_value = chi2.ppf(conf_level, df=len(gaussian_parameters) / 2)

    if fit_param_std is None:
        fit_param_snr_ok = True
        fit_param_snr_detail = "(fit_param_std not provided)"

    elif np.all(fit_param_std > 0):
        fit_param_snr = np.abs(gaussian_parameters) / fit_param_std
        fit_param_snr_ok = bool(np.all(fit_param_snr > _constants.FIT_PARAM_SNR))
        fit_param_snr_detail = f"{fit_param_snr}"

    else:
        fit_param_snr_ok = False
        fit_param_snr_detail = "(fit_param_std is invalid)"

    conds = [
        ((gaussian_parameters > bound[0] + 0.001).all(),
         f"{gaussian_parameters} > {bound[0] + 0.001}",
         "gaussian_parameters > bound[0] + 0.001"),
        ((gaussian_parameters < bound[1] - 0.001).all(),
         f"{gaussian_parameters} < {bound[1] - 0.001}",
         "gaussian_parameters < bound[0] - 0.001"),
        (lower_del_lam <= c1 - c0 <= upper_del_lam,
         f"{lower_del_lam} <= {c1 - c0} <= {upper_del_lam}",
         "lower_del_lam <= c1 - c0 <= upper_del_lam"),
        (sn1 >= sn_line1, f"{sn1} >= {sn_line1}",
         "sn1 >= sn_line1"),
        (sn2 >= sn_line2, f"{sn2} >= {sn_line2}",
         "sn2 >= sn_line2"),
        (np.isfinite(vel1) and np.isfinite(vel2) and abs(vel1 - vel2) <= vmax,
        f"|{vel1} - {vel2}| <= {vmax}",
        f"|vel1 - vel2| <= {vmax}"),
        (min_dr < dr < max_dr, f"{min_dr} < {dr} < {max_dr}",
         "min_dr < dr < max_dr"),
        (ew1_snr > sn_line1, f"{ew1_snr} > {sn_line1}",
         f"ew1_snr > {sn_line1}"),
        (ew2_snr > sn_line2, f"{ew2_snr} > {sn_line2}",
         f"ew2_snr > {sn_line2}"),
         (delta_chi2_line1 > critical_value, f"{delta_chi2_line1} > {critical_value}",
         f"delta_chi2_line1 > {critical_value}"),
        (delta_chi2_line2 > critical_value, f"{delta_chi2_line2} > {critical_value}",
         f"delta_chi2_line2 > {critical_value}"),
        (fit_param_snr_ok, fit_param_snr_detail,
         f"np.all(|fit_params| / fit_param_std > {_constants.FIT_PARAM_SNR})")
    ]

    true_count = sum(c[0] for c in conds)
    false_count = len(conds) - true_count
    result = all(c[0] for c in conds)

    if verbose:
        logger.debug("QSO_INDEX = %s, Condition checks for Z_ABS = %s", qso_id, zabs)
        for status, detail, text in conds:
            logger.debug("%s: %s: %s", text, detail, status)
        logger.debug("Summary: %s / %s conditions satisfied, %s failed.", true_count, len(conds), false_count)
        logger.debug("Final result: %s", result)

    return result

def remove_Mg_falsely_come_from_Fe_absorber(z_after_grouping, lam_obs, residual, error, d_pix, logwave):
    """
    Remove any MgII absorber that arises falsley due to Fe 2586, 2600 doublet,
    i.e., false positive due to Fe lines.

    Args:
        z_after_grouping (list): List of absorbers after grouping.
        lam_obs (numpy.ndarray): Observed wavelengths.
        residual (numpy.ndarray): Residual values.
        error (numpy.ndarray): Error values corresponding to the residuals.
        d_pix (float): Delta pixel value.
        logwave (bool): if wavelength on log scale (default True for SDSS)

    Returns:
        numpy.ndarray: Updated list of absorbers with false positives removed.
    """
    fe1 = 2586.649 #FeII absorption lines
    fe2 = 2600.117
    mg1 = 2796.35
    mg2 = 2803.52

    z = np.array(z_after_grouping)
    nabs = z.size
    match_abs = -1 * np.ones(nabs, dtype='int32')
    w0 = 5*d_pix

    if nabs > 1:
        for p in range(nabs):
            if match_abs[p] != 1:
                lam_fe1_obs = fe1 * (1 + z[p])
                lam_fe2_obs = fe2 * (1 + z[p])

                lam_mgi1_obs_oth = mg1 * (1 + z)
                lam_mgi2_obs_oth = mg2 * (1 + z)

                diff1 = np.abs(lam_fe1_obs - lam_mgi1_obs_oth)
                diff2 = np.abs(lam_fe1_obs - lam_mgi2_obs_oth)
                diff3 = np.abs(lam_fe2_obs - lam_mgi1_obs_oth)
                diff4 = np.abs(lam_fe2_obs - lam_mgi2_obs_oth)

                ind1 = np.where((diff1 > 0) & (diff1 <= w0))[0]
                ind2 = np.where((diff2 > 0) & (diff2 <= w0))[0]
                ind3 = np.where((diff3 > 0) & (diff3 <= w0))[0]
                ind4 = np.where((diff4 > 0) & (diff4 <= w0))[0]
                ind_fin = np.concatenate((ind1, ind2, ind3, ind4))
                n_new = ind_fin.size

                if n_new > 0:
                    lam_rest = lam_obs / (1 + z[p])
                    sn_fe1, sn_fe2 = estimate_snr_for_lines(fe1, fe2, None, None, lam_rest, residual, error, logwave)
                    for k in range(n_new):
                        lam_rest = lam_obs / (1 + z[ind_fin[k]])
                        sn_mg1, sn_mg2 = estimate_snr_for_lines(mg1, mg2, None, None, lam_rest, residual, error, logwave)
                        if (sn_fe1 > sn_mg1) or (sn_fe1 > sn_mg2) or (sn_fe2 > sn_mg1) or (sn_fe2 > sn_mg2):
                            match_abs[ind_fin[k]] = 1  # False positive MgII absorber due to FeII lines
                        else:
                            continue
                else:
                    continue
            else:
                continue
    else:
        match_abs[0] = -1

    return match_abs

def z_abs_from_same_metal_absorber(first_list_z, lam_obs, residual, error, d_pix, use_kernel, logwave):
    """
    Remove any absorber that arises due to the MgII2803 or CIV1550 line but has already
    been detected for the MgII2796 or CIV1548 line, exploiting the doublet property of MgII/CIV to
    remove false positives.

    Args:
        first_list_z (list): List of absorbers after grouping.
        lam_obs (numpy.ndarray): Observed wavelengths.
        residual (numpy.ndarray): Residual values.
        error (numpy.ndarray): Error values corresponding to the residuals.
        d_pix (float): Pixel distance for line separation during Gaussian fitting.
        use_kernel (str, optional): Kernel type (MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, NaI).
        logwave (bool): if wavelength bins are on log scale

    Returns:
        numpy.ndarray: Updated list of absorbers with false positives removed.
    """
    mg1, mg2 = lines[doublet_keys[use_kernel][0]], lines[doublet_keys[use_kernel][1]]

    z = np.array(first_list_z)
    nabs = z.size
    match_abs = -1 * np.ones(nabs)
    w0 = 5*d_pix

    if nabs > 1:
        for p in range(nabs):
            if match_abs[p] != 1:
                lam1_obs = mg1 * (1 + z[p])
                lam2_obs = mg2 * (1 + z)
                diff1 = np.abs(lam1_obs - lam2_obs)
                ind_del = np.where((diff1 > 0) & (diff1 <= w0))[0]

                if ind_del.size > 0:
                    lam_rest = lam_obs / (1 + z[p])
                    sn_mg0, _ = estimate_snr_for_lines(mg1, mg2, None, None, lam_rest, residual, error, logwave)

                    for k in ind_del:
                        lam_rest1 = lam_obs / (1 + z[k])
                        sn_mg1, _ = estimate_snr_for_lines(mg1, mg2, None, None, lam_rest1, residual, error, logwave)

                        if sn_mg0 > sn_mg1:
                            match_abs[k] = 1  # Matched within limits, not true absorbers
    else:
        match_abs[0] = -1

    return match_abs

def contiguous_pixel_remover(abs_z, sn1_all, sn2_all, use_kernel, fitted_params):
    """
    Remove contiguous or duplicate absorbers by evaluating the signal-to-noise ratio (SNR)
    and comparing their line centers.

    Args:
        abs_z (list or numpy.ndarray): List of absorber redshifts.
        sn1_all (list or numpy.ndarray): List of SNR values for the first line.
        sn2_all (list or numpy.ndarray): List of SNR values for the second line.
        use_kernel (str, optional): Kernel type (MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, NaI).
        fitted_params (list of arrays): corresponding gaussian fitting parameters for those redshifts

    Returns:
        list: Updated list of indices indicating bad (1) or good (-1) absorbers.
    """
    # Define constants based on the kernel type
    c0, c1 = lines[doublet_keys[use_kernel][0]], lines[doublet_keys[use_kernel][1]]

    thresh = (c1 - c0) / c0

    abs_z = np.array(abs_z)
    sn1_all = np.array(sn1_all)
    sn2_all = np.array(sn2_all)
    nabs = abs_z.size
    ind_true = np.ones(nabs, dtype='int32')  # Initialize all as bad (1)

    if nabs > 1:
        for k in range(nabs):
            if ind_true[k] == -1:  # Skip if already marked as good
                continue

            # Calculate differences between current absorber and others
            diff = np.abs(abs_z[k] - abs_z)
            ix = np.where((diff >= 0) & (diff <= thresh))[0]

            if ix.size > 0:
                # Consider the current absorber and its closely spaced ones
                candidates = np.append(k, ix)
                best_idx = candidates[0]  # Default to the current one

                # Compare SNR and line positions to decide the best absorber
                for j in candidates:
                    line_diff = abs(fitted_params[j][1] - c0)
                    if line_diff < abs(fitted_params[best_idx][1] - c0):
                        best_idx = j
                    elif line_diff == abs(fitted_params[best_idx][1] - c0):
                        if sn1_all[j] > sn1_all[best_idx]:
                            best_idx = j

                # Mark the best one as good (-1) and others as bad (1)
                ind_true[best_idx] = -1
                ind_true[candidates[candidates != best_idx]] = 1
            else:
                # No closely spaced absorbers, mark the current one as good (-1)
                ind_true[k] = -1
    else:
        ind_true[0] = -1

    return ind_true

@jit(nopython=True)
def redshift_estimate(fitted_obs_l1, fitted_obs_l2, std_fitted_obs_l1, std_fitted_obs_l2, line1, line2):
    """
    Estimate the redshift and its error from Gaussian fitting parameters for
    two spectral lines.

    Args:
        fitted_obs_l1 (float): Gaussian fitted line centre 1 (obs frame).
        fitted_obs_l2 (float): Gaussian fitted line centre 2 (obs frame).
        std_fitted_obs_l1 (float): error on Gaussian fitted line centre 1 (obs frame).
        std_fitted_obs_l2 (float): error on Gaussian fitted line centre 2 (obs frame).
        line1 (float): first line centre of metal spectral line.
        line2 (float): second line centre of metal spectral line.

    Returns:
        tuple: A tuple containing:
            - z_corr (float): mean redshift estimated from the Gaussian fitting parameters.
            - z_err (float): Estimated error in the corrected redshift.
    """

    z1 = (fitted_obs_l1 / line1) - 1
    # define from first line and redshift (more stable and correct)
    fitted_obs_l2 = fitted_obs_l1 + (line2 - line1) * (1 + z1)
    z2 = fitted_obs_l2 / line2 - 1

    err1 = (std_fitted_obs_l1 / line1)
    err2 = (std_fitted_obs_l2 / line2)

    # New redshifts computed using line centers
    # of the first and second Gaussian using a weighted mean

    w1 = line1 / (line1 + line2)
    w2 = line2 / (line1 + line2)
    z_corr = w1 * z1 + w2 * z2
    z_err = np.sqrt((w1 * err1)**2 + (w2 * err2)**2)

    return z_corr, z_err

def return_search_window_wavelength_range(absorber, start_rest_wave=None, end_rest_wave=None, verbose=False):

    """
    Return default red and blue end rest-frame quasar emission wavelength range
    that will be used to define the search windohe given absorber.

    Args:
        absorber (str): Absorber name (e.g., MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, NaI.)
        start_rest_wave (float, optional): start wave in QSO rest-frame for absorber search (default None)
        end_rest_wave (float, optional): end wave in QSO rest-frame for absorber search (default None)

    Returns:
        (blue_end, red_end)
    """

    if start_rest_wave is not None and end_rest_wave is not None:
        if verbose:
            logger.info('using user-defined wavelength search window')
        lam_blue = start_rest_wave
        lam_red = end_rest_wave
    else:
        if verbose:
            logger.info('using default wavelength search window')
        if absorber in ['MgII', 'FeII']:
            lam_blue = lines['CIV_1549']
            lam_red = lines['MgII_2799']

        elif absorber == 'CIV':
            lam_blue = _constants.LAM_CIV_MIN
            lam_red = lines['CIV_1549']

        elif absorber == 'OVI':
            lam_blue = _constants.SMALL_WAVE
            lam_red = lines['OVI_1033']

        elif absorber == 'NV':
            lam_blue = lines['Lya']
            lam_red = lines['NV_1240']

        elif absorber == 'SiIV':
            lam_blue = lines['Lya']
            lam_red = lines['SiIV_1399']

        elif absorber == 'AlIII':
            lam_blue = lines['CIV_1549']
            lam_red = lines['AlIII_1857']

        elif absorber == 'NaI':
            lam_blue = lines['Lya']
            lam_red = lines['NaI_5897']

        elif absorber == 'CaII':
            lam_blue = lines['Lya']
            lam_red = lines['CaII_3969']
        else:
            raise ValueError(f"Unsupported absorber, it must be from {doublet_keys.keys()}")

    return lam_blue, lam_red

def get_search_limits(absorber, zqso, min_wave, max_wave, start_rest_wave=None, end_rest_wave=None, dv=5000, lam_edge_sep=0, verbose=False):
    """
    Return observed-frame wavelength range (lam_start, lam_end) to search for the given absorber.

    Args:
        absorber (str): Absorber name (e.g., MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, NaI.)
        zqso (float): Quasar emission redshift
        min_wave (float): minimum Observed wavelength
        max_wave (float): maximum Observed wavelength
        start_rest_wave (float, optional): start wave in QSO rest-frame for absorber search (default None)
        end_rest_wave (float, optional): end wave in QSO rest-frame for absorber search (default None)
        dv (float): absolute velocity offset from QSO redshift (default 5000 km/s)
        lam_edge_sep (float): Padding in angstroms to avoid edge artifacts

    Returns:
        lam_start, lam_end (float): Observed-frame wavelength limits

    Note:
        ``constants.SMALL_WAVE`` and ``constants.LARGE_WAVE`` act as hard observed-frame
        bounds applied on top of the per-spectrum ``min_wave``/``max_wave`` clip.  Their
        defaults (10 and 1e6) impose no restriction; set them in your constants file to
        restrict the search to a fixed detector range (e.g. ``SMALL_WAVE = 3600``,
        ``LARGE_WAVE = 10000`` for SDSS/DESI optical coverage).
    """

    # Convert velocity offset to redshift offset
    dz = (abs(dv) / speed_of_light) * (1 + zqso)

    lam_blue, lam_red = return_search_window_wavelength_range(absorber, start_rest_wave, end_rest_wave, verbose=verbose)

    if verbose:
        logger.info('wavelength search window in quasar-rest frame: %s Angstroms', (lam_blue, lam_red))

    lam_blue_obs = lam_blue * (1 + zqso + dz)
    lam_red_obs = lam_red * (1 + zqso - dz)

    lam_start = max(min_wave, lam_blue_obs, _constants.SMALL_WAVE) + lam_edge_sep
    lam_end = min(max_wave, lam_red_obs, _constants.LARGE_WAVE) - lam_edge_sep

    if verbose and _constants.SMALL_WAVE > 0:
        logger.info('SMALL_WAVE observed-frame lower limit applied: %.1f Ang', _constants.SMALL_WAVE)
    if verbose and _constants.LARGE_WAVE < 1e9:
        logger.info('LARGE_WAVE observed-frame upper limit applied: %.1f Ang', _constants.LARGE_WAVE)

    return lam_start, lam_end

def absorber_search_window(wavelength, residual, err_residual, zqso, absorber, min_wave, max_wave, start_rest_wave=None, end_rest_wave=None, dv=5000, lam_edge_sep=0, logwave=False, verbose=False):
    """
    Wrapper function to return the most basic wavelength window for absorber
    search.

    Args:
        wavelength (numpy.ndarray): The wavelength array of the QSO spectrum.
        residual (numpy.ndarray): The residual array of the QSO spectrum.
        err_residual (numpy.ndarray): The error residual array of the QSO spectrum.
        zqso (float): The redshift of the QSO.
        absorber (str): (Options: MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, NaI)
        min_wave (float): minimum observed wavelength edge (in Ang)
        max_wave (float): maximum observed wavelength edge (in Ang)
        start_rest_wave (float, optional): start wave in QSO rest-frame for absorber search (default None)
        end_rest_wave (float, optional): end wave in QSO rest-frame for absorber search (default None)
        dv (float): absolute velocity offset from QSO redshift (default 5000 km/s)
        lam_edge_sep (float): separation from minimum/maximum wavelength, i.e. lam_min +/- lam_edge_sep, this is just to make sure that we avoid the very edge of the spectrum
        logwave (bool, optional): If True, wavelength pixels are on a fixed log-scale (e.g. SDSS/DESI). Used to report pixel count. Default is False.
        verbose (bool, optional): If True will print time info. Default is False.

    Returns:
        tuple: A tuple containing unmasked wavelength, residual, and errors.
    """
    start = time.time()

    lam_start, lam_end = get_search_limits(absorber, zqso, min_wave, max_wave, start_rest_wave=start_rest_wave, end_rest_wave=end_rest_wave, dv=dv, lam_edge_sep=lam_edge_sep, verbose=verbose)

    imp_ind = np.where((wavelength >= lam_start) & (wavelength <= lam_end))[0]
    lam_search = wavelength[imp_ind]
    residual = residual[imp_ind]
    error_residual = err_residual[imp_ind]

    ind_mask_ivar = np.isnan(error_residual)
    residual = residual[~ind_mask_ivar]
    lam_search = lam_search[~ind_mask_ivar]
    error_residual = error_residual[~ind_mask_ivar]

    if absorber == 'CIV':
        c_z = 1 + zqso
        # OI 1302 and SiII 1304 masking
        rmv_lam0_1 = (lam_search >= 1296 * c_z) & (lam_search <= _constants.LAM_CIV_MIN * c_z)
        lam_search = lam_search[~rmv_lam0_1]
        error_residual = error_residual[~rmv_lam0_1]
        residual = residual[~rmv_lam0_1]

    if absorber == 'MgII':
        c_z = 1 + zqso
        # Masking other lines (CaII, OH NaD)
        rmv_lam0_1 = (lam_search >= 3928 * c_z) & (lam_search <= 3980 * c_z)
        lam_search = lam_search[~rmv_lam0_1]
        error_residual = error_residual[~rmv_lam0_1]
        residual = residual[~rmv_lam0_1]

    # Masking other lines (OH or atomospheric sky lines)
    rmv_lam0 = (lam_search >= 5568) & (lam_search <= 5588) | \
               (lam_search >= 6295) & (lam_search <= 6305)

    lam_search = lam_search[~rmv_lam0]
    residual = residual[~rmv_lam0]
    error_residual = error_residual[~rmv_lam0]

    if verbose:
        npix = lam_search.size
        ls, le = float(lam_start), float(lam_end)
        if logwave:
            logger.info("search window: %.1f - %.1f Ang, %d pixels (log-lambda scale)", ls, le, npix)
        else:
            dlam = float(lam_search[-1] - lam_search[0]) if npix > 1 else 0.0
            logger.info("search window: %.1f - %.1f Ang, %d pixels (delta-lambda = %.2f Ang)", ls, le, npix, dlam)

    return lam_search, residual, error_residual

def return_if_absorber_can_be_detected_in_a_spectrum(spectra, absorber, **kwargs):
    """Check if an absorber can be searched in a given QSO spectrum.

    This function loads a single QSO spectrum from a spec.QSOSpecRead object,
    removes NaNs, and determines if the absorber's search window
    falls within the spectrum's observed wavelength range.

    Args:
        spectra (object): spec.QSOSpecRead object
        absorber (str): Absorber name (e.g., 'MgII', 'CIV', 'OVI', etc.).
        kwargs (dict): search parameters as described in qsoabsfind.constants()

    Returns:
        tuple: 1 if the absorber can be searched in the spectrum, 0 otherwise and snr value (if snr cut added, otherwise snr value = -1)

    Note:
        - Assumes input spectra are already normalized (flux / continuum).
        - Checks whether enough search window pixels are available after masking NaNs.
        - Users can define 'snr_cut' in kwargs to check if the median SNR > kwargs['snr_cut'] in the absorber search wavelength region.
    """

    start_time = time.time()

    snr_val = -1

    spectra.metadata = Table(spectra.metadata)  # in case spectra.metadata is a Row

    if 'Z' in spectra.metadata.colnames:
        spectra.metadata.rename_column('Z', 'Z_QSO')

    z_qso = spectra.metadata['Z_QSO']
    lam_obs = spectra.wavelength

    if lam_obs.size <= _constants.MIN_NPIXEL:
        return int(0), snr_val

    # Define the wavelength range for searching the absorber
    min_wave, max_wave = lam_obs.min(), lam_obs.max()

    # Retrieve flux and error data, ensuring consistent dtype for Numba compatibility
    residual = spectra.flux.astype('float64')
    error = spectra.error.astype('float64')
    lam_obs = lam_obs.astype('float64')

    # Remove NaN values from the arrays
    non_nan_indices = ~np.isnan(residual)
    lam_obs = lam_obs[non_nan_indices]
    residual = residual[non_nan_indices]
    error = error[non_nan_indices]

    # Identify the wavelength region for searching the specified absorber
    lam_search, unmsk_residual, unmsk_error = absorber_search_window(
        lam_obs, residual, error, z_qso, absorber, min_wave, max_wave,
        lam_edge_sep=kwargs["lam_edge_sep"],
        start_rest_wave=kwargs["start_rest_wave"], end_rest_wave=kwargs["end_rest_wave"],
        dv=kwargs["dv"], logwave=kwargs.get("logwave", False), verbose=kwargs["verbose"]
    )

    # Verify that the arrays are of equal size
    assert lam_search.size == unmsk_residual.size == unmsk_error.size, \
        "Mismatch in array sizes of lam_search, unmsk_residual, and unmsk_error"

    if kwargs.get("verbose", False):
        logger.info('Time took to find available search pixels: %.3f [sec]', time.time()-start_time)

    if lam_search.size <= _constants.MIN_NPIXEL:
        return int(0), snr_val

    if "snr_cut" in kwargs and kwargs["snr_cut"] is not None:
        if "statistics" in kwargs and kwargs["statistics"] is not None:
            stat = kwargs["statistics"]
            if stat == "median":
                snr_val = np.nanmedian(unmsk_residual / unmsk_error)
            elif stat == "mean":
                snr_val = np.nanmean(unmsk_residual / unmsk_error)
            elif isinstance(stat, (int, float)):
                # stat is a percentile (0-100): fraction `stat`% of pixels must have SNR > snr_cut
                pixel_snr = unmsk_residual / unmsk_error
                snr_val = np.nanpercentile(pixel_snr, 100.0 - stat)

            if kwargs.get("verbose", False):
                logger.info('Checking SNR in the wavelength search region %s SNR = %.2f, threshold = %s)', snr_val, kwargs["snr_cut"], stat)
        if snr_val < kwargs["snr_cut"]:
            return int(0), snr_val

    return int(1), snr_val


def _check_searchable_one(params):
    """Worker helper for find_searchable_qsos -- must be module-level to be picklable."""
    from .datamodel import QSOSpecRead
    fits_file, idx, absorber, kwargs = params
    spec = QSOSpecRead(fits_file, index=idx, autoload=True, verbose=False)
    is_good_qso, snr_val = return_if_absorber_can_be_detected_in_a_spectrum(spec, absorber, **kwargs)
    return idx, is_good_qso, snr_val


def find_searchable_qsos(fits_file, absorber, constant_file, ncpus=4, n_qso=None):
    """Run searchability checks for all QSO spectra in parallel.

    For each spectrum, determines whether the given absorber can be searched
    based on the wavelength coverage and pixel count thresholds defined in the
    user constants file.  Overridable package constants (``SMALL_WAVE``,
    ``LARGE_WAVE``, ``LAM_CIV_MIN``, ``MIN_NPIXEL``) are patched from the
    user constants file before the checks run, exactly as done in the main
    convolution pipeline.

    Args:
        fits_file (str): Path to the FITS file containing normalised QSO spectra.
        absorber (str): Absorber name (e.g. ``'MgII'``, ``'CIV'``, ``'OVI'``, ``'NV'``, ``'SiIV'``, ``'AlIII'``, ``'FeII'``, ``'CaII'``, ``'NaI'``).
        constant_file (str): Path to the user constants ``.py`` file.
        ncpus (int): Number of parallel worker processes (default 4).
        n_qso (int or str, optional): Number of spectra to check, or a range
            string such as ``'1-1000'`` or ``'1-1000:10'``.  If ``None``, all
            spectra in the file are checked.

    Returns:
        astropy.table.Table: Table with two columns:

        - ``QSO_INDEX`` (int): Spectrum index in the FITS file.
        - ``IS_GOOD`` (bool): ``True`` if the absorber can be searched in
          that spectrum, ``False`` otherwise.
        - ``SNR`` (float): SNR value in the absorber search region (if snr_cut added, otherwise snr value = -1)
    """
    import os
    import multiprocessing
    from multiprocessing import Pool
    from tqdm import tqdm
    from .config import load_constants
    from .utils import read_nqso_from_header, parse_qso_sequence

    # Load and apply user constants (same override logic as in main pipeline)
    const_path = os.path.abspath(constant_file)
    user_constants = load_constants(const_path)

    print('INFO: Physical constant resolution (user file overrides shown with *):')
    for _name in _constants.OVERRIDABLE_CONSTANTS:
        _user_val = getattr(user_constants, _name, None)
        _pkg_val = getattr(_constants, _name)
        if _user_val is not None and _user_val != _pkg_val:
            print('INFO: %-15s = %s  (overrides package default: %s)' % (_name, _user_val, _pkg_val))
            setattr(_constants, _name, _user_val)
        else:
            print('INFO: %-15s = %s  (package default)' % (_name, _pkg_val))

    # Build per-spectrum kwargs from the user constants search parameters
    search_params = dict(user_constants.search_parameters)

    # Resolve QSO index range
    if n_qso is None:
        n_qso = read_nqso_from_header(fits_file)
    spec_indices = parse_qso_sequence(str(n_qso))

    params_list = [(fits_file, idx, absorber, search_params) for idx in spec_indices]
    n_jobs = min(ncpus, max(1, multiprocessing.cpu_count() - 1))
    print('INFO: Checking searchability of %d spectra for %s absorber using %d CPUs' % (
                len(spec_indices), absorber, n_jobs))

    with Pool(processes=n_jobs) as pool:
        results = list(
            tqdm(
                pool.imap(_check_searchable_one, params_list),
                total=len(params_list),
                desc=f'{absorber} searchability check',
                unit='spec',
            )
        )

    qso_indices, is_good_qso, snr_val = zip(*results) if results else ([], [], [])

    out = Table()
    out['QSO_INDEX'] =  np.asarray(list(qso_indices), dtype=np.int32)
    out['IS_GOOD'] = np.asarray([bool(v) for v in is_good_qso], dtype=bool)
    out['SNR'] = np.asarray(snr_val, dtype=np.float32)

    n_good = sum(out['IS_GOOD'])
    print('INFO: %d / %d QSOs have a searchable %s window' % (n_good, len(out), absorber))

    return out
