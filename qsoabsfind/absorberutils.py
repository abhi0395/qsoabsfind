"""
This script contains a function to find metal absorbers in QSO spectra.
"""

import numpy as np
from .config import speed_of_light, lines
from .utils import elapsed
from numba import jit

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
def check_error_on_residual(l1, l2, lam_rest, residual, error, log):
    """
    Check the error on residuals around specified wavelengths.

    Args:
        l1 (float): First wavelength to check around.
        l2 (float): Second wavelength to check around.
        lam_rest (numpy.ndarray): Rest-frame wavelengths.
        residual (numpy.ndarray): Residual flux values.
        error (numpy.ndarray): Error values corresponding to the residuals.
        log (bool): if wavelength bins are on log scale

    Returns:
        tuple: Mean signal-to-noise ratios (SNR) around the specified wavelengths.
               Returns (mean_sn1, mean_sn2).
    """
    dpix = 5

    if log:
        delta1 = np.abs(l1 * (10**(dpix * 0.0001) - 1))
        delta2 = np.abs(l2 * (10**(dpix * 0.0001) - 1))
    else:
        delta1= lam_rest[1]-lam_rest[0]
        delta2 = delta1

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

@jit(nopython=False)
def group_contiguous_pixel(data, resi, avg):
    """
    Arrange data into groups where successive elements differ by less than a
    given average difference.

    Args:
        data (numpy.ndarray): Array of absorber values for each spectrum.
        resi (numpy.ndarray): Corresponding residual values.
        avg (float): Average difference threshold for grouping contiguous pixels.

    Returns:
        tuple: A tuple containing:
            - list of grouped data
            - list of grouped residuals
    """
    ind_srt = np.argsort(data)
    sorted_data = data[ind_srt]
    sorted_resi = resi[ind_srt] if isinstance(resi, np.ndarray) else None

    groups = [[sorted_data[0]]]
    resi_groups = [[sorted_resi[0]]] if sorted_resi is not None else None

    for i in range(1, len(sorted_data)):
        if sorted_data[i] - groups[-1][-1] < avg:
            groups[-1].append(sorted_data[i])
            if resi_groups is not None:
                resi_groups[-1].append(sorted_resi[i])
        else:
            groups.append([sorted_data[i]])
            if resi_groups is not None:
                resi_groups.append([sorted_resi[i]])

    if resi_groups is not None:
        return groups, resi_groups
    else:
        return groups

@jit(nopython=True)
def weighted_mean(z_values, residuals, gamma):
    """
    Calculate the weighted mean of z_values using residuals raised to the
    power of gamma.

    Args:
        z_values (numpy.ndarray): Array of z values.
        residuals (numpy.ndarray): Array of residual values.
        gamma (int): Exponent for weighting.

    Returns:
        float: Weighted mean of z values.
    """
    weights = 1 / residuals**gamma
    weighted_z = (z_values / residuals**gamma)
    return np.nansum(weighted_z) / np.nansum(weights)

def group_and_weighted_mean_selection_function(master_list_of_pot_absorber, residual, gamma=4):
    """
    Perform grouping, splitting, and median selection from the list of all
    potentially identified absorbers.

    Args:
        master_list_of_pot_absorber (list): List of potential absorbers identified for each spectrum.
        residual (numpy.ndarray): Residual values corresponding to the absorbers.
        gamma (int, optional): Exponent for weighting. Default is 4.

    Returns:
        list: List of unique absorbers for each spectrum.
    """
    if len(master_list_of_pot_absorber) <= 1:
        return master_list_of_pot_absorber

    z_ind = []  # Final list of median redshifts for each spectrum
    abs_list = np.array(master_list_of_pot_absorber)
    z_check = np.log10(1 + abs_list)

    # Grouping contiguous pixels separated by ~0.0006 on redshift scale
    z_temp, res_temp = group_contiguous_pixel(z_check, residual, avg=0.0006)
    for group_z, group_res in zip(z_temp, res_temp):
        if isinstance(group_res, list):
            group_res, group_z = np.array(group_res), np.array(group_z)
        if group_res.size > 0 and np.all(group_res != 0):
            abs_z = weighted_mean(10**group_z - 1, group_res, gamma)
            z_ind.append(abs_z)

    return z_ind

def median_selection_after_combining(combined_final_our_z, lam_search, residual, d_pix=0.6, gamma=4, use_kernel='MgII'):
    """
    Perform grouping and weighted mean from the list of all potentially
    identified absorbers after combining from all the runs with different
    kernel widths.

    Args:
        combined_final_our_z (list): List of potential absorbers identified for each spectrum.
        lam_search (numpy.ndarray): Wavelength search array.
        residual (numpy.ndarray): Residual values corresponding to the absorbers.
        d_pix (float): pixel separation for toloerance in wavelength (default 0.6 A)
        gamma (int): power for lambda to use in 1/lam**gamma weighting scheme (default 4)
        use_kernel (str, optional): Kernel type (MgII). Default is 'MgII'.

    Returns:
        list: List after grouping contiguous pixels for each spectrum.
    """

    if use_kernel=='MgII':thresh=lines['MgII_2796']
    if use_kernel=='CIV':thresh=['CIV_1548']

    z_ind = []  # Final list of median redshifts for each spectrum

    if len(combined_final_our_z) > 1:
        abs_list = np.array(combined_final_our_z)

        # Grouping contiguous pixels separated by ~0.001 on redshift scale
        z_temp = group_contiguous_pixel(abs_list, resi=None, avg=0.001)

        for group in z_temp:
            temp_z = []
            res_z = []
            for zabs in group:
                lam_abs = lam_search / (1 + zabs)
                ind_z = np.where((lam_abs >= thresh-d_pix) & (lam_abs <= thresh+d_pix))[0]

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

#@jit(nopython=False)
def remove_Mg_falsely_identified_as_Fe_absorber(index, z_after_grouping, lam_obs, residual, error, d_pix):
    """
    Remove any MgII absorber that arises falsley due to Fe 2586, 2600 doublet,
    i.e., false positive due to Fe lines.

    Args:
        z_after_grouping (list): List of absorbers after grouping.
        lam_obs (numpy.ndarray): Observed wavelengths.
        residual (numpy.ndarray): Residual values.
        error (numpy.ndarray): Error values corresponding to the residuals.
        d_pix (float): Delta pixel value.

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
                    sn_fe1, sn_fe2 = check_error_on_residual(fe1, fe2, lam_rest, residual, error)
                    for k in range(n_new):
                        lam_rest = lam_obs / (1 + z[ind_fin[k]])
                        sn_mg1, sn_mg2 = check_error_on_residual(mg1, mg2, lam_rest, residual, error)
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

#@jit(nopython=False)
def z_abs_from_same_metal_absorber(first_list_z, lam_obs, residual, error, d_pix=0.6, use_kernel='MgII'):
    """
    Remove any absorber that arises due to the MgII2803 or CIV1550 line but has already
    been detected for the MgII2796 or CIV1548 line, exploiting the doublet property of MgII/CIV to
    remove false positives.

    Args:
        first_list_z (list): List of absorbers after grouping.
        lam_obs (numpy.ndarray): Observed wavelengths.
        residual (numpy.ndarray): Residual values.
        error (numpy.ndarray): Error values corresponding to the residuals.
        d_pix (float): Pixel distance for line separation during Gaussian fitting. Default is 0.6.
        use_kernel (str, optional): Kernel type (MgII, CIV). Default is 'MgII'.

    Returns:
        numpy.ndarray: Updated list of absorbers with false positives removed.
    """
    if use_kernel=='MgII':
        mg1, mg2 = lines['MgII_2796'], lines['MgII_2803']

    if use_kernel=='CIV':
        mg1, mg2 = lines['CIV_1548'], lines['CIV_1550']

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
                    sn_mg0, _ = check_error_on_residual(mg1, mg2, lam_rest, residual, error)

                    for k in ind_del:
                        lam_rest1 = lam_obs / (1 + z[k])
                        sn_mg1, _ = check_error_on_residual(mg1, mg2, lam_rest1, residual, error)

                        if sn_mg0 > sn_mg1:
                            match_abs[k] = 1  # Matched within limits, not true absorbers
    else:
        match_abs[0] = -1

    return match_abs

#@jit(nopython=False)
def contiguous_pixel_remover(abs_z, sn1_all, sn2_all, use_kernel='MgII'):
    """
    Remove contiguous pixels by evaluating the signal-to-noise ratio (SNR)
    for absorbers.

    Args:
        abs_z (list or numpy.ndarray): List of absorber redshifts.
        sn1_all (list or numpy.ndarray): List of SNR values for the first line.
        sn2_all (list or numpy.ndarray): List of SNR values for the second line.
        use_kernel (str, optional): Kernel type (MgII, CIV). Default is 'MgII'.

    Returns:
        list: Updated list of absorbers with false positives removed.
    """
    if use_kernel=='MgII':
        thresh = (lines['MgII_2803']-lines['MgII_2796'])/lines['MgII_2796']
    if use_kernel=='CIV':
        thresh = (lines['CIV_1550']-lines['CIV_1548'])/lines['CIV_1548']
    abs_z = np.array(abs_z)
    sn1_all = np.array(sn1_all)
    sn2_all = np.array(sn2_all)
    nabs = abs_z.size
    ind_true = -1 * np.ones(nabs, dtype='int32')

    if nabs > 1:
        for k in range(nabs):
            diff = np.abs(abs_z[k] - abs_z)
            ix = np.where((diff > 0) & (diff <= thresh))[0]
            if ix.size > 0:
                sn1_temp = sn1_all[ix]

                for j in range(ix.size):
                    if sn1_all[k] > sn1_temp[j]:
                        ind_true[k] = -1
                    else:
                        ind_true[k] = 1
            else:
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

    err1 = (std_fitted_obs_l1 / line1) * z1
    err2 = (std_fitted_obs_l2 / line2) * z2

    z_corr = 0.5 * (z1 + z2)  # New redshifts computed using line centers of the first and second Gaussian
    z_err = np.sqrt(0.25 * (err1**2 + err2**2))

    return z_corr, z_err


def absorber_search_window(wavelength, residual, err_residual, zqso, absorber, min_wave, max_wave, verbose=False):
    """
    Wrapper function to return the most basic wavelength window for absorber
    search.

    Args:
        wavelength (numpy.ndarray): The wavelength array of the QSO spectrum.
        residual (numpy.ndarray): The residual array of the QSO spectrum.
        err_residual (numpy.ndarray): The error residual array of the QSO spectrum.
        zqso (float): The redshift of the QSO.
        absorber (str): Options 'CIV', 'MgII'.
        min_wave (float): minimum wavelength edge (in Ang)
        max_wave (float): maximum wavelength edge (in Ang)
        verbose (bool, optional): If True will print time info. Default is False.

    Returns:
        tuple: A tuple containing unmasked wavelength, residual, and errors.
    """
    start = elapsed(None, "")

    if absorber == 'MgII':
        lam_CIV = lines['CIV_1549'] * (1 + zqso + lines['dz_start']) #redshifted from CIV emission lines
        lam_MgII = lines['MgI_2799'] * (1 + zqso - lines['dz_end']) #blueshifted MgII emission lines
        lam_start = max(min_wave, lam_CIV)
        lam_end = min(max_wave, lam_MgII)
    elif absorber == 'CIV':
        dz = (lines['dv'] / speed_of_light) * (1 + zqso)
        lam_CIV = lines['CIV_1549'] * (1 + zqso + dz)
        lam_start = max(min_wave, 1310 * (1 + zqso))  # This is from Cooksey et al 2013
        lam_end = min(lam_CIV, max_wave)
    else:
        raise ValueError("Absorber must be 'CIV' or 'MgII'")

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
