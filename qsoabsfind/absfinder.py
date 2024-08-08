"""
This script contains a function to run convolution based absorber finder on a single spectrum.
"""

import numpy as np
from functools import reduce
from operator import add
from .utils import convolution_fun, vel_dispersion, elapsed
from .absorberutils import (
    estimate_local_sigma_conv_array, group_and_weighted_mean_selection_function,
    median_selection_after_combining,
    remove_Mg_falsely_come_from_Fe_absorber, z_abs_from_same_metal_absorber,
    contiguous_pixel_remover, estimate_snr_for_lines, absorber_search_window,
    find_valid_indices
)
from .ew import measure_absorber_properties_double_gaussian
from .config import load_constants
from .spec import QSOSpecRead
import time

constants = load_constants()
lines, oscillator_parameters, speed_of_light = constants.lines, constants.oscillator_parameters, constants.speed_of_light

def read_single_spectrum_and_find_absorber(fits_file, spec_index, absorber, **kwargs):
    """
    This function retrieves a single QSO spectrum from a FITS file, processes the data to remove NaNs,
    and prepares the spectrum for absorber search within specified wavelength regions
    and runs the convolution based adaptive S/N method to detect absorbers in the spectrum.

    Args:
        fits_file (str): Path to the FITS file containing normalized QSO spectra.
                         The file must include extensions for FLUX, ERROR, WAVELENGTH
                         and METADATA which must contain keyword Z_QSO.
        spec_index (int): Index of the quasar spectrum to retrieve from the FITS file.
        absorber (str): Name of the absorber to search for (e.g., 'MgII', 'CIV').
        kwargs (dict): search parameters as described in qsoabsfind.constants()

    Returns:
        tuple: Contains lists of various parameters related to detected absorbers.
            - index (list): QSO spec index searched
            - zabs (list of floats): redshifts of absorbers detected
            - params (list of arrays): gaussian fit parameters for each absorber
            - errror params (list of arrays): errors on gaussian fit parameters for each absorber
            - EW1 (list of floats): Equivalent width of line 1 for each absorber
            - EW2 (list of floats): Equivalent width of line 2 for each absorber
            - EW total (list of floats): Total Equivalent width of line 1 and line 2 for each absorber
            - errors EW1 (list of floats): errors on Equivalent width of line 1 for each absorber
            - errors EW2 (list of floats): errors on Equivalent width of line 2 for each absorber
            - errors EW total (list of floats): errors on Total Equivalent width of line 1 and line 2 for each absorber

    Raises:
        AssertionError: If the sizes of `lam_search`, `unmsk_residual`, and `unmsk_error` do not match.

    Note:
        - This function assumes that the input spectra are already normalized (i.e., flux divided by continuum).
        - The wavelength search region is determined dynamically based on the observed wavelength range.
    """
    start_time = time.time()
    # Read the specified QSO spectrum from the FITS file
    spectra = QSOSpecRead(fits_file, spec_index)
    z_qso = spectra.metadata['Z_QSO']
    lam_obs = spectra.wavelength

    # Define the wavelength range for searching the absorber
    min_wave, max_wave = lam_obs.min() + kwargs["lam_edge_sep"], lam_obs.max() - kwargs["lam_edge_sep"]  # avoiding edges

    # Retrieve flux and error data, ensuring consistent dtype for Numba compatibility
    residual, error = spectra.flux.astype('float64'), spectra.error.astype('float64')
    lam_obs = lam_obs.astype('float64')

    # Remove NaN values from the arrays
    non_nan_indices = ~np.isnan(residual)
    lam_obs, residual, error = lam_obs[non_nan_indices], residual[non_nan_indices], error[non_nan_indices]

    # Identify the wavelength region for searching the specified absorber
    lam_search, unmsk_residual, unmsk_error = absorber_search_window(
        lam_obs, residual, error, z_qso, absorber, min_wave, max_wave, verbose=kwargs['verbose'])

    # Verify that the arrays are of equal size
    assert lam_search.size == unmsk_residual.size == unmsk_error.size, "Mismatch in array sizes of lam_search, unmsk_residual, and unmsk_error"

    kwargs.pop("lam_edge_sep") # just remove this keyword as its not used the following function.

    (index_spec, pure_z_abs, pure_gauss_fit, pure_gauss_fit_std, pure_ew_first_line_mean, pure_ew_second_line_mean, pure_ew_total_mean, pure_ew_first_line_error, pure_ew_second_line_error, pure_ew_total_error, redshift_err, sn1_all, sn2_all) = convolution_method_absorber_finder_in_QSO_spectra(spec_index, absorber, lam_obs, residual, error, lam_search, unmsk_residual, unmsk_error, **kwargs)

    # Print progress for every spectrum processed
    elapsed(start_time, f"\nTime taken to finish absorber detection for index = {spec_index} is: ")

    return (index_spec, pure_z_abs, pure_gauss_fit, pure_gauss_fit_std, pure_ew_first_line_mean, pure_ew_second_line_mean, pure_ew_total_mean, pure_ew_first_line_error, pure_ew_second_line_error, pure_ew_total_error, redshift_err, sn1_all, sn2_all)


def convolution_method_absorber_finder_in_QSO_spectra(spec_index, absorber='MgII', lam_obs=None, residual=None, error=None,
lam_search=None, unmsk_residual=None, unmsk_error=None, ker_width_pixels=[3, 4, 5, 6, 7, 8], coeff_sigma=2.5,
mult_resi=1, d_pix=0.6, pm_pixel=200, sn_line1=3, sn_line2=2, use_covariance=False, resolution=69, logwave=True, verbose=False):
    """
    Detect absorbers with doublet properties in SDSS quasar spectra using a
    convolution method. This function identifies potential absorbers based on
    user-defined threshold criteria, applies Gaussian fitting to reject false
    positives, and computes the equivalent widths (EWs) of the lines, returning
    the redshifts, EWs, and fitting parameters.

    Args:
        spec_index (int): Index of quasar in the spectra 2D array.
        absorber (str): Absorber name for searching doublets (MgII, CIV). Default is 'MgII'.
        lam_obs (numpy.array): observed wavelength array.
        residual (numpy.array): residual (i.e. flux/continuum) array
        error (numpy.array): error on residuals
        lam_search (numpy.array): search observed wavelength array (i.e. region where absorber will be looked for).
        unmsk_residual (numpy.array): search residual array (residuals at search wavelength pixels)
        unmsk_error (numpy.array): error on residuals array in search wavelength region
        ker_width_pix (list): List of kernel widths in pixels. Default is [3, 4, 5, 6, 7, 8].
        coeff_sigma (float): Coefficient for sigma to apply threshold in the convolved array. Default is 2.5.
        mult_resi (float): Factor to shift the residual up or down. Default is 1.
        d_pix (float): Pixel distance for line separation during Gaussian fitting. Default is 0.6.
        pm_pixel (int): Pixel parameter for local noise estimation (default 200).
        sn_line1 (float): Signal-to-noise ratio for thresholding for line1 (default 3).
        sn_line2 (float): Signal-to-noise ratio for thresholding for line2 (default 3).
        use_covariance (bool): if want to use full covariance of scipy curvey_fit for EW error calculation (default is False)
        resolution (float): wavelength resolution of spectrum (in km/s), e.g. SDSS: ~69, DESI: ~70 (also defined in constants)
        logwave (bool): if wavelength on log scale (default True for SDSS)
        verbose (bool): if want to print a lot of outputs for debugging (default False)

    Returns:
        tuple: Contains lists of various parameters related to detected absorbers.
            - index (list): QSO spec index searched
            - zabs (list): redshifts of absorbers detected
            - params (list of arrays): gaussian fit parameters for each absorber
            - errror params (list of arrays): errors on gaussian fit parameters for each absorber
            - EW1 (list): Equivalent width of line 1 for each absorber
            - EW2 (list): Equivalent width of line 2 for each absorber
            - EW total (list): Total Equivalent width of line 1 and line 2 for each absorber
            - errors EW1 (list): errors on Equivalent width of line 1 for each absorber
            - errors EW2 (list): errors on Equivalent width of line 2 for each absorber
            - errors EW total (list): errors on Total Equivalent width of line 1 and line 2 for each absorber
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

    line_sep = line2 - line1

    del_sigma = line1 * resolution / speed_of_light  # in Ang

    bd_ct, x_sep = 2.5, 10 # multiple for bound definition (for line centres and widths of line)

    # bounds for gaussian fitting, to avoid very bad candidates
    bound = ((np.array([2e-2, line1 - bd_ct * d_pix, del_sigma-0.1, 2e-2, line2 - bd_ct * d_pix, del_sigma-0.1])),
             (np.array([1.11, line1 + bd_ct * d_pix, x_sep * del_sigma+0.1, 1.11, line2 + bd_ct * d_pix, x_sep * del_sigma+0.1])))

    # line separation tolerance (fitted line centers should not be outside, centre +/- d_pix)
    lower_del_lam = line_sep - d_pix
    upper_del_lam = line_sep + d_pix

    # Kernel width computation
    width_kernel = np.array([ker * resolution * ((f1 * line1 + f2 * line2) / (f1 + f2)) / (speed_of_light * 2.35) for ker in ker_width_pixels])

    combined_final_our_z = []

    if len(unmsk_residual) > 0:
        for sig_ker in width_kernel:
            line_centre = (line1 + line2) / 2

            conv_arr = convolution_fun(absorber, mult_resi * unmsk_residual, sig_ker, amp_ratio=0.5, log=logwave, index=spec_index)
            sigma_cr = estimate_local_sigma_conv_array(conv_arr, pm_pixel=pm_pixel)
            thr = np.nanmedian(conv_arr) - coeff_sigma * sigma_cr

            conv_arr[np.isnan(conv_arr)] = 1e5
            our_z_ind = conv_arr < thr
            conv_arr[conv_arr == 1e5] = np.nan

            our_z = lam_search[our_z_ind] / line_centre - 1
            residual_our_z = unmsk_residual[our_z_ind]

            new_our_z, new_res_arr = find_valid_indices(our_z, residual_our_z, lam_search, conv_arr, sigma_cr, coeff_sigma, d_pix, f1 / f2, line1, line2, logwave)

            final_our_z = group_and_weighted_mean_selection_function(new_our_z, np.array(new_res_arr))

            combined_final_our_z.append(final_our_z)

        combined_final_our_z = reduce(add, combined_final_our_z)
        combined_final_our_z = list(set(combined_final_our_z))
        combined_final_our_z = median_selection_after_combining(combined_final_our_z, lam_obs, residual)

        combined_final_our_z = np.array(combined_final_our_z)
        combined_final_our_z = combined_final_our_z[~np.isnan(combined_final_our_z)]
        combined_final_our_z = combined_final_our_z.tolist()

        if len(combined_final_our_z)>0:

            z_abs, z_err, fit_param, fit_param_std, EW_first_line_mean, EW_second_line_mean, EW_total_mean, EW_first_line_error, EW_second_line_error, EW_total_error = measure_absorber_properties_double_gaussian(
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
            sn1_all = np.zeros(len(z_abs))
            sn2_all = np.zeros(len(z_abs))


            for m in range(len(z_abs)):
                if len(fit_param[m]) > 0 and not np.all(np.isnan(fit_param[m])):

                    z_new, z_new_error, fit_param_temp, fit_param_std_temp, EW_first_temp_mean, EW_second_temp_mean, EW_total_temp_mean, EW_first_error_temp, EW_second_error_temp, EW_total_error_temp = measure_absorber_properties_double_gaussian(
                        index=spec_index, wavelength=lam_obs, flux=residual, error=error, absorber_redshift=[z_abs[m]], bound=bound, use_kernel=absorber, d_pix=d_pix)

                    if len(fit_param_temp[0]) > 0 and not np.all(np.isnan(fit_param_temp[0])):
                        gaussian_parameters = np.array(fit_param_temp[0])
                        lam_rest = lam_obs / (1 + z_abs[m])
                        c0 = gaussian_parameters[1]
                        c1 = gaussian_parameters[4]
                        #S/N estimation
                        sn1, sn2 = estimate_snr_for_lines(c0, c1, lam_rest, residual, error, logwave)
                        # resolution corrected velocity dispersion (should be greater than 0)
                        vel1, vel2 = vel_dispersion(c0, c1, gaussian_parameters[2], gaussian_parameters[5], resolution)

                        if (gaussian_parameters > bound[0] + 0.01).all() and (gaussian_parameters < bound[1] - 0.01).all() and lower_del_lam <= c1 - c0 <= upper_del_lam and sn1 > sn_line1 and sn2 > sn_line2 and vel1 > 0 and vel2 > 0:
                            pure_z_abs[m] = z_new
                            pure_gauss_fit[m] = fit_param_temp[0]
                            pure_gauss_fit_std[m] = fit_param_std_temp[0]
                            pure_ew_first_line_mean[m] = EW_first_temp_mean[0]
                            pure_ew_second_line_mean[m] = EW_second_temp_mean[0]
                            pure_ew_total_mean[m] = EW_total_temp_mean[0]
                            pure_ew_first_line_error[m] = EW_first_error_temp[0]
                            pure_ew_second_line_error[m] = EW_second_error_temp[0]
                            pure_ew_total_error[m] = EW_total_error_temp[0]
                            redshift_err[m] = z_new_error
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
            sn1_all = sn1_all[valid_indices]
            sn2_all = sn2_all[valid_indices]
            if len(pure_z_abs) > 0:
                if absorber=='MgII':
                    match_abs1 = remove_Mg_falsely_come_from_Fe_absorber(spec_index, pure_z_abs, lam_obs, residual, error, d_pix, logwave)
                else:
                    match_abs1 = -1*np.ones(len(pure_z_abs))
                match_abs2 = z_abs_from_same_metal_absorber(pure_z_abs, lam_obs, residual, error, d_pix, absorber, logwave)
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
                redshift_err = np.array([0])
                pure_z_abs = np.array([0])
                pure_gauss_fit = pure_gauss_fit_std = np.array([[0, 0, 0, 0, 0, 0]])
                pure_ew_first_line_mean = pure_ew_second_line_mean = pure_ew_total_mean = np.array([0])
                pure_ew_first_line_error = pure_ew_second_line_error = pure_ew_total_error = np.array([0])
                sn1_all = sn2_all = np.array([0])

            not_found = max(1, len(pure_z_abs))
            index_spec = [spec_index for _ in range(not_found)]
            return (index_spec, pure_z_abs.tolist(), pure_gauss_fit.tolist(), pure_gauss_fit_std.tolist(), pure_ew_first_line_mean.tolist(), pure_ew_second_line_mean.tolist(), pure_ew_total_mean.tolist(),
                    pure_ew_first_line_error.tolist(), pure_ew_second_line_error.tolist(), pure_ew_total_error.tolist(), redshift_err.tolist(), sn1_all.tolist(), sn2_all.tolist())
        else:
            return ([spec_index], [0], [[0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0]], [0], [0], [0], [0], [0], [0], [0], [0], [0])
    else:
        return ([spec_index], [0], [[0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0]], [0], [0], [0], [0], [0], [0], [0], [0], [0])
