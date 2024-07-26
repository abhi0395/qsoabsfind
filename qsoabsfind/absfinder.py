import numpy as np
from functools import reduce
from operator import add
from numba import jit, njit, prange
from .utils import convolution_fun
from .absorberutils import (
    estimate_local_sigma_conv_array, group_and_weighted_mean_selection_function,
    median_selection_after_combining, measure_absorber_properties_double_gaussian,
    remove_Mg_falsely_identified_fe_absberber, z_abs_from_same_metal_absorber,
    contiguous_pixel_remover, check_error_on_residual, redshift_estimate, absorber_search_window
)
from .constants import lines, search_parameters, speed_of_light, oscillator_parameters
from .spec import QSOSpecRead

@njit
def find_valid_indices(our_z, residual_our_z, lam_search, conv_arr, sigma_cr, coeff_sigma, d_pix, beta, line1, line2):
    """
    Identify valid absorber redshifts based on convolution thresholds.

    This function evaluates each candidate redshift in the `our_z` array to check if it meets the
    secondary threshold criteria for valid absorbers. The secondary thresholds are based on the
    convolution array values around expected absorber wavelengths.

    Parameters:
    - our_z (numpy.ndarray): Array of candidate redshift values.
    - residual_our_z (numpy.ndarray): Array of residual flux values corresponding to the candidate redshifts.
    - lam_search (numpy.ndarray): Wavelength array in the observed frame.
    - conv_arr (numpy.ndarray): Convolution array used for initial redshift detection.
    - sigma_cr (numpy.ndarray): Array of local sigma values for noise estimation.
    - coeff_sigma (float): Coefficient for sigma to apply secondary threshold.
    - d_pix (float): Pixel distance for line separation during Gaussian fitting.
    - beta (float): Ratio of oscillator strengths of the doublet lines.
    - line1 (float): Wavelength of the first line of the doublet in the rest frame.
    - line2 (float): Wavelength of the second line of the doublet in the rest frame.

    Returns:
    - new_our_z (list): List of validated redshift values.
    - new_res_arr (list): List of corresponding residual flux values for validated redshifts.
    """
    new_our_z = []
    new_res_arr = []

    for k in range(1, len(our_z)):
        lam_check = line1 - 5 * d_pix * (1 + our_z[k])
        lam_check1 = line2 + 5 * d_pix * (1 + our_z[k])
        lam_check_thresh = d_pix * 3
        ind_check = (lam_search >= lam_check - lam_check_thresh) & (lam_search <= lam_check + lam_check_thresh)
        ind_check1 = (lam_search >= lam_check1 - lam_check_thresh) & (lam_search <= lam_check1 + lam_check_thresh)

        if not (np.all(np.isnan(conv_arr[ind_check])) and np.all(np.isnan(conv_arr[ind_check1]))):
            conv_arr1 = conv_arr[ind_check]
            conv_arr2 = conv_arr[ind_check1]

            sec_thr1 = np.nanmedian(conv_arr) - coeff_sigma / beta * sigma_cr[ind_check]
            sec_thr2 = np.nanmedian(conv_arr) - coeff_sigma / beta * sigma_cr[ind_check1]

            if np.all(conv_arr1 <= sec_thr1) or np.all(conv_arr2 <= sec_thr2):
                new_our_z.append(our_z[k])
                new_res_arr.append(residual_our_z[k])

    return new_our_z, new_res_arr

def convolution_method_absorber_finder_in_QSO_spectra(fits_file, spec_index, absorber='MgII', ker_width_pix=[3, 4, 5, 6, 7, 8], coeff_sigma=2.5, mult_resi=1, d_pix=0.6, pm_pixel=200, sn_line1=3, sn_line2=2, use_covariance=False):
    """
    Detect absorbers with doublet properties in SDSS quasar spectra using a convolution method. This function identifies
    potential absorbers based on user-defined threshold criteria, applies Gaussian fitting to reject false positives,
    and computes the equivalent widths (EWs) of the lines, returning the redshifts, EWs, and fitting parameters.

    Parameters:
    fits_file (str): Path to the FITS file containing Normalized QSO spectra
    (i.e. flux/continuum), must contain FLUX, ERROR (i.e. error/continuum),
    WAVELENGTH, and TGTDETAILS extensions.
    In TGTDETAILS, must contain keywords like RA_QSO, DEC_QSO, Z_QSO.

    spec_index (int or numpy.array): Index of quasars in the data matrix.
    absorber (str): Absorber name for searching doublets (MgII, CIV). Default is 'MgII'.
    ker_width_pix (list): List of kernel widths in pixels. Default is [3, 4, 5, 6, 7, 8].
    coeff_sigma (float): Coefficient for sigma to apply threshold in the convolved array. Default is 2.5.
    mult_resi (float): Factor to shift the residual up or down. Default is 1.
    d_pix (float): Pixel distance for line separation during Gaussian fitting. Default is 0.6.
    pm_pixel (int): Pixel parameter for local noise estimation (default 200).
    sn_line1 (float): Signal-to-noise ratio for thresholding for line1.
    sn_line2 (float): Signal-to-noise ratio for thresholding for line2.
    use_covariance (bool): if want to use full covariance of scipy curvey_fit for EW error calculation (default is False)

    Returns:
    tuple: Contains lists of various parameters related to detected absorbers.
    """

    # Constants
    if absorber == 'MgII':
        line1, line2 = lines['MgII_2796'], lines['MgII_2803']
        f1, f2 = oscillator_parameters['MgII_f1'], oscillator_parameters['MgII_f2']
    elif absorber == 'CIV':
        line1, line2 = lines['CIV_1548'], lines['CIV_1550']
        f1, f2 = oscillator_parameters['CIV_f1'], oscillator_parameters['CIV_f2']
    else:
        raise ValueError(f"No support for {absorber}, only supports MgII and CIV")

    spectra = QSOSpecRead(fits_file, spec_index)
    z_qso = spectra.tgtdetails['Z_QSO']
    lam_obs = spectra.wavelength
    min_wave, max_wave = lam_obs.min() + 500, lam_obs.max() - 500

    line_sep = line2 - line1
    resolution = 69  # km/s for SDSS or DESI

    delta_z_start = lines['dz_start']
    delta_z_end = lines['dz_end']

    del_sigma = line1 * resolution / speed_of_light  # in Ang

    bound = ((np.array([2e-2, line1 - d_pix, del_sigma, 2e-2, line2 - d_pix, del_sigma])),
             (np.array([1.11, line1 + d_pix, 10 * del_sigma, 1.11, line2 + d_pix, 10 * del_sigma])))

    # Print progress
    if spec_index % 5000 == 0:
        print(f'Detection finished up to spec index = {spec_index}', flush=True)

    # Retrieve quasar data
    residual, error = spectra.flux, spectra.error
    ind_nonan = ~np.isnan(residual)
    residual, error = residual[ind_nonan], error[ind_nonan]
    lam_obs = lam_obs[ind_nonan]

    lam_search, unmsk_residual, unmsk_error = absorber_search_window(
        lam_obs, residual, error, z_qso, absorber, min_wave, max_wave, verbose=False)

    # Kernel width computation
    width_kernel = np.array([ker * resolution * ((f1 * line1 + f2 * line2) / (f1 + f2)) / (speed_of_light * 2.35) for ker in ker_width_pix])

    combined_final_our_z = []

    if len(unmsk_residual) > 0:
        for sig_ker in width_kernel:
            line_centre = (line1 + line2) / 2
            conv_arr = convolution_fun(absorber, mult_resi * unmsk_residual, sig_ker, amp_ratio=0.5, log=True)
            sigma_cr = estimate_local_sigma_conv_array(conv_arr, pm_pixel=pm_pixel)
            thr = np.nanmedian(conv_arr) - coeff_sigma * sigma_cr
            conv_arr[np.isnan(conv_arr)] = 1e5
            our_z_ind = conv_arr < thr
            conv_arr[conv_arr == 1e5] = np.nan

            our_z = lam_search[our_z_ind] / line_centre - 1
            residual_our_z = unmsk_residual[our_z_ind]

            new_our_z, new_res_arr = find_valid_indices(our_z, residual_our_z, lam_search, conv_arr, sigma_cr, coeff_sigma, d_pix, f1 / f2, line1, line2)

            final_our_z = group_and_weighted_mean_selection_function(new_our_z, np.array(new_res_arr))
            combined_final_our_z.append(final_our_z)

        combined_final_our_z = reduce(add, combined_final_our_z)
        combined_final_our_z = list(set(combined_final_our_z))
        combined_final_our_z = median_selection_after_combining(combined_final_our_z, lam_obs, residual)

        combined_final_our_z = np.array(combined_final_our_z)
        combined_final_our_z = combined_final_our_z[~np.isnan(combined_final_our_z)]
        combined_final_our_z = combined_final_our_z.tolist()

        z_abs, fit_param, fit_param_std, EW_first_line_mean, EW_second_line_mean, EW_total_mean, EW_first_line_error, EW_second_line_error, EW_total_error = measure_absorber_properties_double_gaussian(
            index=spec_index, wavelength=lam_obs, flux=residual, error=error, absorber_redshift=combined_final_our_z, bound=bound, use_kernel=absorber, d_pix=d_pix, use_covariance=use_covariance)

        pure_z_abs = np.zeros(len(z_abs))
        pure_gauss_fit = np.zeros((len(z_abs), 6))
        pure_gauss_fit_std = np.zeros((len(z_abs), 6))
        pure_ew_first_line_mean = np.zeros(len(z_abs))
        pure_ew_second_line_mean = np.zeros(len(z_abs))
        pure_ew_total_mean = np.zeros(len(z_abs))
        pure_ew_first_line_error = np.zeros(len(z_abs))
        pure_ew_second_line_error = np.zeros(len(z_abs))
        pure_ew_total_error = np.zeros(len(z_abs))
        redshift_err = np.zeros(len(z_abs))
        final_redshift = np.zeros(len(z_abs))
        sn1_all = np.zeros(len(z_abs))
        sn2_all = np.zeros(len(z_abs))

        for m in prange(len(z_abs)):
            if len(fit_param[m]) > 0 and not np.all(np.isnan(fit_param[m])):
                gauss_params = np.array(fit_param[m])
                std_params = np.array(fit_param_std[m])

                lower_del_lam = line_sep - d_pix
                upper_del_lam = line_sep + d_pix

                new_z_after_fit, z_err = redshift_estimate(gauss_params, std_params, line1, line2)

                z_abs_temp, fit_param_temp, fit_param_std_temp, EW_first_temp_mean, EW_second_temp_mean, EW_total_temp_mean, EW_first_error_temp, EW_second_error_temp, EW_total_error_temp = measure_absorber_properties_double_gaussian(
                    index=spec_index, wavelength=lam_obs, flux=residual, error=error, absorber_redshift=[new_z_after_fit], bound=bound, use_kernel=absorber, d_pix=d_pix)

                if len(fit_param_temp[0]) > 0 and not np.all(np.isnan(fit_param_temp[0])):
                    gaussian_parameters = np.array(fit_param_temp[0])
                    lam_rest = lam_obs / (1 + z_abs_temp[0])
                    c0 = gaussian_parameters[1]
                    c1 = gaussian_parameters[4]
                    sn1, sn2 = check_error_on_residual(c0, c1, lam_rest, residual, error)

                    if (gaussian_parameters > bound[0] + 0.01).all() and (gaussian_parameters < bound[1] - 0.01).all() and lower_del_lam <= c1 - c0 <= upper_del_lam and sn1 > sn_line1 and sn2 > sn_line2:
                        pure_z_abs[m] = z_abs_temp[0]
                        pure_gauss_fit[m] = fit_param_temp[0]
                        pure_gauss_fit_std[m] = fit_param_std_temp[0]
                        pure_ew_first_line_mean[m] = EW_first_temp_mean[0]
                        pure_ew_second_line_mean[m] = EW_second_temp_mean[0]
                        pure_ew_total_mean[m] = EW_total_temp_mean[0]
                        pure_ew_first_line_error[m] = EW_first_error_temp[0]
                        pure_ew_second_line_error[m] = EW_second_error_temp[0]
                        pure_ew_total_error[m] = EW_total_error_temp[0]
                        redshift_err[m] = z_err
                        final_redshift[m] = z_abs[m]
                        sn1_all[m] = sn1
                        sn2_all[m] = sn2

        valid_indices = pure_z_abs != 0

        pure_z_abs = pure_z_abs[valid_indices]
        pure_gauss_fit = pure_gauss_fit[valid_indices]
        pure_gauss_fit_std = pure_gauss_fit_std[valid_indices]
        pure_ew_first_line_mean = pure_ew_first_line_mean[valid_indices]
        pure_ew_second_line_mean = pure_ew_second_line_mean[valid_indices]
        pure_ew_total_mean = pure_ew_total_mean[valid_indices]
        pure_ew_first_line_error = pure_ew_first_line_error[valid_indices]
        pure_ew_second_line_error = pure_ew_second_line_error[valid_indices]
        pure_ew_total_error = pure_ew_total_error[valid_indices]
        redshift_err = redshift_err[valid_indices]
        final_redshift = final_redshift[valid_indices]
        sn1_all = sn1_all[valid_indices]
        sn2_all = sn2_all[valid_indices]

        if len(pure_z_abs) > 0:
            match_abs1 = remove_Mg_falsely_identified_fe_absberber(spec_index, pure_z_abs, lam_obs, residual, error, use_kernel=absorber, d_pix=d_pix)
            match_abs2 = z_abs_from_same_metal_absorber(pure_z_abs, lam_obs, residual, error, use_kernel=absorber)
            ind_z = contiguous_pixel_remover(pure_z_abs, sn1_all, sn2_all)
            sel_indices = (match_abs1 == -1) & (match_abs2 == -1) & (ind_z == -1)  # pure final absorber candidates

            # Select final quantities based on sel_indices
            pure_z_abs = pure_z_abs[sel_indices]
            pure_gauss_fit = pure_gauss_fit[sel_indices]
            pure_gauss_fit_std = pure_gauss_fit_std[sel_indices]
            pure_ew_first_line_mean = pure_ew_first_line_mean[sel_indices]
            pure_ew_second_line_mean = pure_ew_second_line_mean[sel_indices]
            pure_ew_total_mean = pure_ew_total_mean[sel_indices]
            pure_ew_first_line_error = pure_ew_first_line_error[sel_indices]
            pure_ew_second_line_error = pure_ew_second_line_error[sel_indices]
            pure_ew_total_error = pure_ew_total_error[sel_indices]
            redshift_err = redshift_err[sel_indices]
            sn1_all = sn1_all[sel_indices]
            sn2_all = sn2_all[sel_indices]
        else:
            redshift_err = [0]
            pure_z_abs = [0]
            pure_gauss_fit, pure_gauss_fit_std = [[0, 0, 0, 0, 0, 0]]
            pure_ew_first_line_mean = pure_ew_second_line_mean = pure_ew_total_mean = [0]
            pure_ew_first_line_error = pure_ew_second_line_error = pure_ew_total_error = [0]
            sn1_all = sn2_all = [0]

        not_found = max(1, len(final_redshift))
        index_spec = [spec_index for _ in range(not_found)]

        return (index_spec, pure_z_abs.tolist(), pure_gauss_fit.tolist(), pure_gauss_fit_std.tolist(), pure_ew_first_line_mean.tolist(), pure_ew_second_line_mean.tolist(), pure_ew_total_mean.tolist(),
                pure_ew_first_line_error.tolist(), pure_ew_second_line_error.tolist(), pure_ew_total_error.tolist(), redshift_err.tolist(), sn1_all.tolist(), sn2_all.tolist())
    else:
        return ([spec_index], [0], [[0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0]], [0], [0], [0], [0], [0], [0], [0], [0], [0])
