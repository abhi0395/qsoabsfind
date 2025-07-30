"""
This script contains a function to find metal absorbers in QSO spectra.
"""

import numpy as np
from numba import jit
from .config import load_constants
from .utils import elapsed

# Constants
constants = load_constants()
lines = constants.lines
speed_of_light = constants.speed_of_light
doublet_keys = constants.doublet_keys

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
    npix = 3 # number of pixels around a line minima
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
def calculate_doublet_ratio(ew1, ew2, ew1_error, ew2_error):
    """
    Calculate the doublet ratio and its associated error.

    Args:
        ew1 (float): Equivalent width of the first line.
        ew2 (float): Equivalent width of the second line.
        ew1_error (float): Error associated with the first equivalent width.
        ew2_error (float): Error associated with the second equivalent width.

    Returns:
        tuple: A tuple containing:
            - doublet_ratio (float): The ratio of ew1 to ew2.
            - doublet_ratio_error (float): The propagated error for the doublet ratio.
    """
    # Calculate the doublet ratio
    doublet_ratio = ew1 / ew2

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
        tuple: Mean signal-to-noise ratios (SNR) around the specified wavelengths.
               Returns (mean_sn1, mean_sn2).
    """
    if sig1 is None or sig2 is None:
        dpix = 5
        if log:
            delta1 = np.abs(l1 * (10**(dpix * 0.0001) - 1))
            delta2 = np.abs(l2 * (10**(dpix * 0.0001) - 1))
        else:
            delta1 = dpix * (lam_rest[1]-lam_rest[0])
            delta2 = delta1
    else:
        nsig = 4 # for gaussian more than 99.7 percentile data is within 4sigma
        delta1, delta2 = nsig * sig2, nsig * sig2

    ind1 = np.where((lam_rest > l1 - delta1) & (lam_rest < l1 + delta1))[0]
    ind2 = np.where((lam_rest > l2 - delta2) & (lam_rest < l2 + delta2))[0]

    resi1 = residual[ind1]
    resi2 = residual[ind2]

    err1 = error[ind1]
    err2 = error[ind2]

    median = 1  # Assuming median residual value is 1

    diff1 = np.abs(median - resi1)
    diff2 = np.abs(median - resi2)

    sum_diff1 = np.nansum(diff1)
    sum_diff2 = np.nansum(diff2)
    sum_err1 = np.sqrt(np.nansum(err1**2))
    sum_err2 = np.sqrt(np.nansum(err2**2))


    mean_sn1, mean_sn2 = -1, -1 # in case failure

    if sum_err1 != 0 and sum_err2 !=0:
        mean_sn1 = sum_diff1 / sum_err1
        mean_sn2 = sum_diff2 / sum_err2

    return mean_sn1, mean_sn2

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

def group_and_select_weighted_redshift(redshifts, fluxes, delta_z):
    """
    Group contiguous redshifts and select the highest weighted
    (corresponding to minimum flux) redshift from each group.

    Args:
        redshifts (list or np.array): list of redshifts.
        fluxes (list or np.array): corresponding fluxes near each redshift.
        delta_z (float): the maximum difference between redshifts to consider them contiguous.

    Returns:
        best_redshifts: list of best redshifts from each group.
    """

    # Ensure inputs are numpy arrays for easy manipulation
    redshifts = np.array(redshifts)
    fluxes = np.array(fluxes)

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

def median_selection_after_combining(combined_final_our_z, lam_search, residual, d_pix, use_kernel, delta_z, gamma=4):
    """
    Perform grouping and weighted mean from the list of all potentially
    identified absorbers after combining from all the runs with different
    kernel widths.

    Args:
        combined_final_our_z (list): List of potential absorbers identified for each spectrum.
        lam_search (numpy.ndarray): Wavelength search array.
        residual (numpy.ndarray): Residual values corresponding to the absorbers.
        d_pix (float): pixel separation for toloerance in wavelength (default 0.6 A)
        use_kernel (str, optional): Kernel type.
        delta_z_threshold (float): the maximum difference between redshifts to consider them contiguous.
        gamma (int): power for lambda to use in 1/lam**gamma weighting scheme (default 4)

    Returns:
        list: List after grouping contiguous pixels for each spectrum.
    """
    thresh  = lines[doublet_keys[use_kernel][0]]

    z_ind = []  # Final list of median redshifts for each spectrum
    ct = 2
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
        use_kernel (str, optional): Kernel type (MgII, CIV).
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
        use_kernel (str, optional): Kernel type (MgII, CIV, OVI, NV, SiIV, AlIII, FeII).
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
            ix = np.where((diff > 0) & (diff <= thresh))[0]

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
    z2 = (fitted_obs_l2 / line2) - 1

    err1 = (std_fitted_obs_l1 / line1)
    err2 = (std_fitted_obs_l2 / line2)

    z_corr = 0.5 * (z1 + z2)  # New redshifts computed using line centers of the first and second Gaussian
    z_err = np.sqrt(0.25 * (err1**2 + err2**2))

    return z_corr, z_err

def return_search_window_wavelength_range(absorber):

    """
    Return default red and blue end rest-frame quasar emission wavelength range
    that will be used to define the search windohe given absorber.

    Args:
        absorber (str): Absorber name (e.g., MgII, CIV, OVI, NV, SiIV, AlIII, FeII.)
    Returns:
        (blue_end, red_end)
    """
    if lines["start_rest_wave"] is not None and lines["end_rest_wave"] is not None:
        print('INFO: using user-defined wavelength search window')
        lam_blue = lines["start_rest_wave"]
        lam_red = lines["end_rest_wave"]
    else:
        print('INFO: using default wavelength search window')
        if absorber in ['MgII', 'FeII']:
            lam_blue = lines['CIV_1549']
            lam_red = lines['MgII_2799']

        elif absorber == 'CIV':
            lam_blue = 1310.0
            lam_red = lines['CIV_1549']

        elif absorber == 'OVI':
            # assuming 3600 to be starting wavelength
            # and maximum redshift of quasar to be 6 in SDSS/DESI like spectra
            lam_blue = 3600/(1+6.2)
            lam_red = lines['OVI_1033']

        elif absorber == 'NV':
            lam_blue = lines['Lyb_1026']
            lam_red = lines['NV_1240']

        elif absorber == 'SiIV':
            lam_blue = lines['Lya']
            lam_red = lines['SiIV_1399']

        elif absorber == 'AlIII':
            lam_blue = lines['CIV_1549']
            lam_red = lines['AlIII_1857']

        else:
            raise ValueError(f"Unsupported absorber, it must be from {doublet_keys.keys()}")

    return lam_blue, lam_red


def get_search_limits(absorber, zqso, min_wave, max_wave, lines, lam_edge_sep=0):
    """
    Return observed-frame wavelength range (lam_start, lam_end) to search for the given absorber.

    Args:
        absorber (str): Absorber name (e.g., MgII, CIV, OVI, NV, SiIV, AlIII, FeII.)
        zqso (float): Quasar emission redshift
        min_wave (float): minimum Observed wavelength
        max_wave (float): maximum Observed wavelength
        lam_edge_sep (float): Padding in angstroms to avoid edge artifacts

    Returns:
        lam_start, lam_end (float): Observed-frame wavelength limits
    """

    # Convert velocity offset to redshift offset
    dz = (abs(lines['dv']) / speed_of_light) * (1 + zqso)

    lam_blue, lam_red = return_search_window_wavelength_range(absorber)

    print(f'INFO: wavelength search window in quasar-rest frame: {lam_blue, lam_red} Angstroms')

    lam_blue_obs = lam_blue * (1 + zqso + dz)
    lam_red_obs = lam_red * (1 + zqso - dz)

    lam_start = max(min_wave, lam_blue_obs) + lam_edge_sep
    lam_end = min(max_wave, lam_red_obs) - lam_edge_sep

    return lam_start, lam_end

def absorber_search_window(wavelength, residual, err_residual, zqso, absorber, min_wave, max_wave, lam_edge_sep=0, verbose=False):
    """
    Wrapper function to return the most basic wavelength window for absorber
    search.

    Args:
        wavelength (numpy.ndarray): The wavelength array of the QSO spectrum.
        residual (numpy.ndarray): The residual array of the QSO spectrum.
        err_residual (numpy.ndarray): The error residual array of the QSO spectrum.
        zqso (float): The redshift of the QSO.
        absorber (str): (Options: MgII, CIV, OVI, NV, SiIV, AlIII, FeII)
        min_wave (float): minimum wavelength edge (in Ang)
        max_wave (float): maximum wavelength edge (in Ang)
        lam_edge_sep (float): separation from minimum/maximum wavelength, i.e. lam_min +/- lam_edge_sep, this is just to make sure that we avoid the very edge of the spectrum
        verbose (bool, optional): If True will print time info. Default is False.

    Returns:
        tuple: A tuple containing unmasked wavelength, residual, and errors.
    """
    start = elapsed(None, "")

    lam_start, lam_end = get_search_limits(absorber, zqso, min_wave, max_wave, lines, lam_edge_sep=lam_edge_sep)

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
        rmv_lam0_1 = (lam_search >= 1296 * c_z) & (lam_search <= 1310 * c_z)
        lam_search = lam_search[~rmv_lam0_1]
        error_residual = error_residual[~rmv_lam0_1]
        residual = residual[~rmv_lam0_1]

    # Masking other lines (CaII, OH NaD)
    rmv_lam0 = (lam_search >= 3928) & (lam_search <= 3940) | \
               (lam_search >= 3963) & (lam_search <= 3975) | \
               (lam_search >= 5568) & (lam_search <= 5588) | \
               (lam_search >= 6295) & (lam_search <= 6305)

    lam_search = lam_search[~rmv_lam0]
    residual = residual[~rmv_lam0]
    error_residual = error_residual[~rmv_lam0]

    if verbose:
        elapsed(start, f"INFO: final wave window selection for {absorber} took")

    return lam_search, residual, error_residual

def return_if_absorber_can_be_detected_in_a_spectrum(spectra, absorber, **kwargs):
    """Check if an absorber can be searched in a given QSO spectrum.

    This function loads a single QSO spectrum from a FITS file,
    removes NaNs, and determines if the absorber's search window
    falls within the spectrum's observed wavelength range.

    Args:
        spectra (object): spec.QSOSpecRead object
        absorber (str): Absorber name (e.g., 'MgII', 'CIV', 'OVI', etc.).
        kwargs (dict): search parameters as described in qsoabsfind.constants()

    Returns:
        int: 1 if the absorber can be searched in the spectrum, 0 otherwise.

    Note:
        - Assumes input spectra are already normalized (flux / continuum).
        - Checks whether enough search window pixels are available after masking NaNs.
        - Users can define 'snr_cut' in kwargs to check if the median SNR > kwargs['snr_cut']
        in the absorber search wavelength region.
    """
    import time
    from astropy.table import Table
    start_time = time.time()

    spectra.metadata = Table(spectra.metadata)  # in case spectra.metadata is a Row

    if 'Z' in spectra.metadata.colnames:
        spectra.metadata.rename_column('Z', 'Z_QSO')

    z_qso = spectra.metadata['Z_QSO']
    lam_obs = spectra.wavelength

    if lam_obs.size <= 10:
        return 0

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
        lam_edge_sep=kwargs["lam_edge_sep"], verbose=kwargs["verbose"]
    )

    # Verify that the arrays are of equal size
    assert lam_search.size == unmsk_residual.size == unmsk_error.size, \
        "Mismatch in array sizes of lam_search, unmsk_residual, and unmsk_error"

    print(f'INFO: Time took to find available search pixels: {time.time()-start_time:.3f} [sec]')

    if lam_search.size <= 10:
        return 0

    if "snr_cut" in kwargs and kwargs["snr_cut"] is not None:
        snr_median = np.nanmedian(unmsk_residual / unmsk_error)
        print(f'INFO: Checking SNR in the wavelength search region (median SNR = {snr_median:.2f}, threshold = {kwargs["snr_cut"]})')
        if snr_median < kwargs["snr_cut"]:
            return 0

    return 1
