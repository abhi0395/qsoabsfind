"""
This script contains a function to run the main convolution
based absorber algorithm on a single spectrum.
"""
from functools import reduce
from operator import add
import time
import logging
import numpy as np
from astropy.table import Table
from .utils import convolution_fun, vel_dispersion, snr_of_spectra
from .absorberutils import (
    estimate_local_sigma_conv_array,
    remove_Mg_falsely_come_from_Fe_absorber,
    z_abs_from_same_metal_absorber,
    contiguous_pixel_remover,
    estimate_snr_for_lines,
    absorber_search_window,
    find_valid_indices,
    calculate_doublet_ratio,
    group_and_select_weighted_redshift,
    check_absorber_selection,
)
from .ew import (
    measure_absorber_properties_double_gaussian,
    trapezoidal_ew,
    fit_cost_double_gaussian
)
from .datamodel import QSOSpecRead
from .config import load_constants

# Constants -- imported via the module object so startup-time patches propagate here.
from .constants import lines, oscillator_parameters, speed_of_light, doublet_keys
from . import constants as _constants

logger = logging.getLogger(__name__)

def read_single_spectrum_and_find_absorber(fits_file, spec_index, absorber, constant_file=None, **kwargs):
    """
    This function retrieves a single QSO spectrum from a FITS file, processes the data to remove NaNs,
    and prepares the spectrum for absorber search within specified wavelength regions
    and runs the convolution based adaptive S/N method to detect absorbers in the spectrum.

    Args:
        fits_file (str): Path to the FITS file containing normalized QSO spectra.
                         The file must include extensions for FLUX, ERROR, WAVELENGTH
                         and METADATA which must contain keyword Z_QSO.
        spec_index (int): Index of the quasar spectrum to retrieve from the FITS file.
        absorber (str): Name of the absorber to search for (e.g., MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, NaI).
        constant_file (str, optional): Path to a user constants file. When provided, the file
            is loaded, global constants (``SMALL_WAVE``, ``LARGE_WAVE``, ``LAM_CIV_MIN``,
            ``MIN_NPIXEL``, ``ZABS_KNOWN_MAX_DV``) are patched in-place, and
            ``search_parameters`` from the file are merged into ``kwargs`` as defaults
            (explicit ``kwargs`` take precedence). Default is None.
        kwargs (dict): search parameters as taken in convolution_method..()
            An optional key ``zabs_known`` (float or list) may be provided.
            When present, the absorber search window is skipped and the code
            goes straight to Gaussian fitting at the supplied redshift(s).

    Returns:
        dict: Contains lists of various parameters related to detected absorbers.
            - index_spec (list): QSO spec index searched
            - z_abs (list of floats): redshifts of absorbers detected
            - gauss_fit (list of arrays): gaussian fit parameters for each absorber
            - gauss_fit_std (list of arrays): errors on gaussian fit parameters for each absorber
            - ew_1_mean (list of floats): Equivalent width of line 1 for each absorber
            - ew_2_mean (list of floats): Equivalent width of line 2 for each absorber
            - ew_total_mean (list of floats): Total Equivalent width of line 1 and line 2 for each absorber
            - ew_1_error (list of floats): errors on Equivalent width of line 1 for each absorber
            - ew_2_error (list of floats): errors on Equivalent width of line 2 for each absorber
            - ew_total_error (list of floats): errors on Total Equivalent width of line 1 and line 2 for each absorber
            - z_abs_err (list): errors on redshifts of absorbers detected
            - sn_1 (list): SNR of line 1 for each absorber
            - sn_2 (list): SNR of line 2 for each absorber
            - vel_disp1 (list): rest-frame velocity dispersion of line 1 for each absorber (in km/s)
            - vel_disp2 (list): rest-frame velocity dispersion of line 2 for each absorber (in km/s)
            - delta_chi2_line1 (list): per-line delta_chi2 for line 1
            - delta_chi2_line2 (list): per-line delta_chi2 for line 2

    Raises:
        AssertionError: If the sizes of `lam_search`, `unmsk_residual`, and `unmsk_error` do not match.

    Note:
        - This function assumes that the input spectra are already normalized (i.e., flux divided by continuum).
        - The wavelength search region is determined dynamically based on the observed wavelength range.
    """
    if constant_file is not None:
        _user_constants = load_constants(constant_file)

        # Patch global constants in-place with user overrides.
        for _name in _constants.OVERRIDABLE_CONSTANTS:
            _user_val = getattr(_user_constants, _name, None)
            if _user_val is not None:
                setattr(_constants, _name, _user_val)

        # Merge search_parameters as base; only explicitly provided kwargs override.
        _merged = dict(_user_constants.search_parameters)

        for key, val in kwargs.items():
            if val is not None:
                _merged[key] = val

        kwargs = _merged

    start_time = time.time()
    verbose = kwargs.get('verbose', False)
    # Read the specified QSO spectrum from the FITS file
    if verbose:
        logger.info("Starting search for QSO INDEX = %s", spec_index)
    spectra = QSOSpecRead(fits_file, index=spec_index, autoload=True, verbose=verbose) # verbose=True, shows time

    spectra.metadata = Table(spectra.metadata) # in case spectra.metadata is a Row

    if 'Z' in spectra.metadata.colnames:
        spectra.metadata.rename_column('Z', 'Z_QSO')

    z_qso = spectra.metadata['Z_QSO']
    lam_obs = spectra.wavelength

    # Define the wavelength range for searching the absorber
    min_wave, max_wave = lam_obs.min(), lam_obs.max()

    # Retrieve flux and error data, ensuring consistent dtype for Numba compatibility
    residual, error = spectra.flux.astype('float64'), spectra.error.astype('float64')
    lam_obs = lam_obs.astype('float64')

    # Remove NaN values from the arrays
    non_nan_indices = np.isfinite(residual)
    lam_obs, residual, error = lam_obs[non_nan_indices], residual[non_nan_indices], error[non_nan_indices]

    zabs_known = kwargs.get("zabs_known", None)

    if zabs_known is not None:
        # Known-redshift mode: skip absorber search window entirely.
        # lam_search and unmsk_residual are not needed; the convolution
        # step is bypassed inside convolution_method_absorber_finder_in_QSO_spectra.
        lam_search = None
        unmsk_residual = None
        if verbose:
            logger.info("zabs_known provided, skipping search window for spec index = %s", spec_index)
        snr_val, stat = snr_of_spectra(residual, error, **kwargs)
    else:
        # Identify the wavelength region for searching the specified absorber
        if verbose:
            logger.debug(f"PATCH CHECK before search window: SMALL_WAVE={_constants.SMALL_WAVE} LARGE_WAVE={_constants.LARGE_WAVE}")

        lam_search, unmsk_residual, unmsk_error = absorber_search_window(
            lam_obs, residual, error, z_qso, absorber, min_wave, max_wave, start_rest_wave=kwargs["start_rest_wave"], end_rest_wave=kwargs["end_rest_wave"],
            dv=kwargs["dv"], lam_edge_sep=kwargs["lam_edge_sep"], logwave=kwargs.get("logwave", False), verbose=verbose, qso_dv_mask_emline=kwargs.get("qso_dv_mask_emline", None))

        if verbose:
            logger.debug("lam_search range: %.2f %.2f  npix=%d", np.nanmin(lam_search), np.nanmax(lam_search),lam_search.size)

        assert lam_search.size == unmsk_residual.size == unmsk_error.size, "Mismatch in array sizes of lam_search, unmsk_residual, and unmsk_error"

        snr_cut = kwargs.get("snr_cut")
        snr_val, stat = snr_of_spectra(unmsk_residual, unmsk_error, **kwargs)
        if snr_cut is not None and snr_val < snr_cut:
            if verbose:
                logger.debug("SNR check failed (%s snr_val=%.2f < snr_cut=%.2f), spec index = %s",
                            stat, snr_val, snr_cut, spec_index)
            result = _build_result(
                [spec_index], [-1], [[0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0]], [0], [0], [0],
                [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]
            )
            result[f'snr_qso'] = snr_val
            return result

    not_allowed_args = ["lam_edge_sep", "start_rest_wave", "end_rest_wave",
                            "dv", "lam_red", "lam_blue",
                            "snr_cut", "statistics", "qso_dv_mask_emline", "continuum_error_frac"]

    conv_kwargs = {}
    for key in kwargs.keys():
        if key not in not_allowed_args:
            conv_kwargs[key] = kwargs[key]

    if verbose:
        logger.debug("search absorber = %s", absorber)
        logger.debug("Z_QSO = %s", z_qso[0])

    result = convolution_method_absorber_finder_in_QSO_spectra(
        spec_index,
        absorber,
        lam_obs,
        residual,
        error,
        lam_search,
        unmsk_residual,
        **conv_kwargs,
    )

    result["snr_qso"] = snr_val

    if verbose:
        logger.debug("Time taken to finish %s detection for index = %s Quasar: %.2f seconds", absorber, spec_index, time.time() - start_time)

    return result


def _build_result(index_spec, z_abs, gauss_fit, gauss_fit_std, ew_1_mean, ew_2_mean,
                  ew_total_mean, ew_1_error, ew_2_error, ew_total_error,
                  z_abs_err, sn_1, sn_2, vel_disp1, vel_disp2,
                  delta_chi2_line1, delta_chi2_line2, pure_redchi2,
                  vel_disp1_err=None, vel_disp2_err=None, zabs_known=None):
    result = {
        'index_spec': index_spec,
        'z_abs': z_abs,
        'gauss_fit': gauss_fit,
        'gauss_fit_std': gauss_fit_std,
        'ew_1_mean': ew_1_mean,
        'ew_2_mean': ew_2_mean,
        'ew_total_mean': ew_total_mean,
        'ew_1_error': ew_1_error,
        'ew_2_error': ew_2_error,
        'ew_total_error': ew_total_error,
        'z_abs_err': z_abs_err,
        'sn_1': sn_1,
        'sn_2': sn_2,
        'vel_disp1': vel_disp1,
        'vel_disp2': vel_disp2,
        'vel_disp1_err': vel_disp1_err if vel_disp1_err is not None else [0] * len(vel_disp1),
        'vel_disp2_err': vel_disp2_err if vel_disp2_err is not None else [0] * len(vel_disp2),
        'delta_chi2_line1': delta_chi2_line1,
        'delta_chi2_line2': delta_chi2_line2,
        'pure_redchi2': pure_redchi2
    }
    if zabs_known is not None:
        result['zabs_known'] = zabs_known
    return result


def _get_doublet_constants(absorber):
    # Look up rest wavelengths, oscillator strengths and derived quantities for the doublet.
    # doublet_keys is patched at startup from the user constants file, so custom absorbers
    # registered there are fully supported without any code changes.
    if absorber not in doublet_keys:
        raise ValueError(
            f"Absorber '{absorber}' not found in doublet_keys. "
            f"Built-in absorbers: {list(doublet_keys.keys())}. "
            "To use a custom doublet, add it to your constants file "
            "(see docs/paramfile.rst for the required format)."
        )
    line1 = lines[doublet_keys[absorber][0]]
    line2 = lines[doublet_keys[absorber][1]]
    f1 = oscillator_parameters[f'{absorber}_f1']
    f2 = oscillator_parameters[f'{absorber}_f2']
    prod1 = f1 * line1
    prod2 = f2 * line2
    line_ratio = max(prod1, prod2) / min(prod1, prod2)
    line_sep = line2 - line1
    del_z = line_sep / line1
    return line1, line2, f1, f2, line_ratio, line_sep, del_z


def _compute_resolution(lam_search, lam_obs, line1, logwave,
                        res_wave_start=None, res_val_start=None,
                        res_wave_end=None,   res_val_end=None,
                        res_is_R=True):
    """Compute wavelength sampling and instrumental LSF sigma in velocity.

    Args:
        lam_search (numpy.ndarray): Observed wavelength array used for the
            absorber search.
        lam_obs (numpy.ndarray): Full observed wavelength array in Angstrom.
        line1 (float): Rest-frame wavelength of the first doublet line in
            Angstrom.
        logwave (bool): Whether the wavelength grid is uniform in log10(lambda).
        res_wave_start (float, optional): Blue wavelength anchor in Angstrom.
        res_val_start (float, optional): Resolution value at the blue anchor.
        res_wave_end (float, optional): Red wavelength anchor in Angstrom.
        res_val_end (float, optional): Resolution value at the red anchor.
        res_is_R (bool, optional): If True, res_val_start and res_val_end are
            resolving powers R. If False, they are Gaussian sigma_v values
            in km/s.

    Returns:
        tuple: ``(wave_res, resolution, median_resolution, del_sigma)``.
            ``wave_res`` is Angstrom/pixel for a linear grid and dex/pixel
            for a log10 grid. ``resolution`` is the instrumental Gaussian
            sigma_v in km/s at each wavelength in ``lam_obs``.
            ``median_resolution`` is the median sigma_v over the search
            wavelength range. ``del_sigma`` is the corresponding rest-frame
            Gaussian sigma in Angstrom for ``line1``.
    """
    lam_search = np.asarray(lam_search, dtype=float)
    lam_obs = np.asarray(lam_obs, dtype=float)

    if not logwave:
        wave_res = np.nanmedian(np.diff(lam_search))
        dv_pix = speed_of_light * np.abs(np.gradient(lam_obs)) / lam_obs
    else:
        wave_res = np.nanmedian(np.diff(np.log10(lam_search)))
        dv_pix = np.full_like(lam_obs, speed_of_light * np.log(10.0) * wave_res, dtype=float)

    have_anchors = None not in (res_wave_start, res_val_start, res_wave_end, res_val_end)

    if have_anchors:
        slope = (res_val_end - res_val_start) / (res_wave_end - res_wave_start)
        res_val = res_val_start + slope * (lam_obs - res_wave_start)
        resolution = speed_of_light / (2.355 * res_val) if res_is_R else np.asarray(res_val, dtype=float)
    else:
        logger.warning('No instrumental-resolution anchors supplied; using one spectral pixel as a proxy for sigma_v.')
        resolution = dv_pix

    search_mask = (lam_obs >= np.nanmin(lam_search)) & (lam_obs <= np.nanmax(lam_search)) & np.isfinite(resolution)
    median_resolution = np.nanmedian(resolution[search_mask]) if np.any(search_mask) else np.nanmedian(resolution)
    del_sigma = median_resolution * line1 / speed_of_light

    return wave_res, resolution, median_resolution, del_sigma



def _compute_fit_bounds(line1, line2, line_sep, d_pix, del_sigma):
    # Build the six-parameter Gaussian fitting bounds and the acceptable
    # range for the observed line separation.  bd_ct and x_sep set how far
    # each centre and width is allowed to deviate from the theoretical value.
    bd_ct = _constants.GAUSS_FIT_BD_CT
    x_sep = _constants.GAUSS_FIT_X_SEP
    edge  = _constants.GAUSS_FIT_EDGE

    amp_min = _constants.GAUSS_AMP_MIN   # physical lower limit for absorption depth
    amp_max = _constants.GAUSS_AMP_MAX   # physical upper limit for absorption depth

    bound = (
        np.array([
            amp_min, line1 - bd_ct * d_pix, max(0.1, del_sigma - edge),
            amp_min, line2 - bd_ct * d_pix, max(0.1, del_sigma - edge)
        ]),
        np.array([
            amp_max, line1 + bd_ct * d_pix, x_sep * del_sigma + edge,
            amp_max, line2 + bd_ct * d_pix, x_sep * del_sigma + edge
        ])
    )

    lower_del_lam = line_sep - d_pix
    upper_del_lam = line_sep + d_pix

    return bound, lower_del_lam, upper_del_lam


def _run_convolution_and_find_candidates(absorber, mult_resi, unmsk_residual, residual,
                                         lam_search, lam_obs, width_kernel, pm_pixel,
                                         coeff_sigma, line_ratio, line1, line2, del_z,
                                         logwave, wave_res, f1, f2, spec_index, verbose):
    # Run the matched-filter convolution for every requested kernel width and collect
    # raw candidate redshifts.  After the loop the lists are merged, deduplicated and
    # thinned by median_selection_after_combining so the output is a clean list of
    # distinct candidate redshifts ready for Gaussian fitting.
    combined_final_our_z = []
    for sig_ker in width_kernel:
        if verbose:
            logger.debug("convolving for kernel width: %s Angstrom.", sig_ker)
        line_centre = (line1 + line2) / 2
        conv_arr = convolution_fun(absorber, mult_resi * unmsk_residual, sig_ker,
                                   log=logwave, wave_res=wave_res, index=spec_index, f1=f1, f2=f2)
        sigma_cr = estimate_local_sigma_conv_array(conv_arr, pm_pixel=pm_pixel)
        thr = np.nanmedian(conv_arr) - coeff_sigma * sigma_cr
        conv_arr[np.isnan(conv_arr)] = 1e5
        our_z_ind = conv_arr < thr
        conv_arr[conv_arr == 1e5] = np.nan
        our_z = lam_search[our_z_ind] / line_centre - 1
        residual_our_z = unmsk_residual[our_z_ind]
        if verbose:
            logger.debug("sigma cut on convolved flux for potential candidates")
        new_our_z, new_res_arr = find_valid_indices(our_z, residual_our_z, lam_search, conv_arr,
                                                    sigma_cr, coeff_sigma, line_ratio, line1, line2, logwave)
        final_our_z = group_and_select_weighted_redshift(new_our_z, new_res_arr, residual,
                                                         lam_obs, line1, line2, del_z)
        combined_final_our_z.append(final_our_z)

    if verbose:
        logger.debug("combining redshifts")
    combined_final_our_z = reduce(add, combined_final_our_z)
    combined_final_our_z = list(set(combined_final_our_z))
    if verbose:
        logger.debug("potential candidates before combining: %s", combined_final_our_z)

    combined_final_our_z = np.asarray(combined_final_our_z, dtype=float)
    combined_final_our_z = combined_final_our_z[np.isfinite(combined_final_our_z)]

    if combined_final_our_z.size == 0:
        combined_final_our_z = []
    else:
        combined_final_our_z = np.sort(combined_final_our_z)

        cleaned = [combined_final_our_z[0]]
        min_sep = 0.10 * del_z

        for z in combined_final_our_z[1:]:
            if z - cleaned[-1] > min_sep:
                cleaned.append(z)

        combined_final_our_z = cleaned

    if verbose:
        logger.debug("potential candidates after combining: %s", combined_final_our_z)
    return combined_final_our_z


def _validate_candidates(spec_index, z_abs_candidates, lam_obs, residual, error, bound,
                         absorber, d_pix, f1, f2, resolution, line_ratio,
                         lower_del_lam, upper_del_lam, sn_line1, sn_line2,
                         logwave, use_covariance, nboot, conf_level, verbose,
                         trapz_ew_sigma=None):
    # For each candidate redshift, re-run the double-Gaussian fit, compute SNR,
    # velocity dispersion and doublet ratio, then keep only those that pass
    # check_absorber_selection.  All output arrays are indexed the same way as
    # z_abs_candidates so the caller can apply a boolean mask afterwards.
    z_abs, _, fit_param, _, _, _, _, _, _, _, _, _ = measure_absorber_properties_double_gaussian(
        index=spec_index, wavelength=lam_obs, flux=residual, error=error,
        absorber_redshift=z_abs_candidates, bound=bound, use_kernel=absorber,
        d_pix=d_pix, use_covariance=use_covariance, nboot=nboot)

    n = len(z_abs)
    pure_z_abs = np.zeros(n)
    pure_gauss_fit = np.zeros((n, 6))
    pure_gauss_fit_std = np.zeros((n, 6))
    pure_ew_first_line_mean = np.zeros(n)
    pure_ew_second_line_mean = np.zeros(n)
    pure_ew_total_mean = np.zeros(n)
    pure_ew_first_line_error = np.zeros(n)
    pure_ew_second_line_error = np.zeros(n)
    pure_ew_total_error = np.zeros(n)
    redshift_err = np.zeros(n)
    sn1_all = np.zeros(n)
    sn2_all = np.zeros(n)
    vel_disp1 = np.zeros(n)
    vel_disp2 = np.zeros(n)
    vel_disp1_err = np.zeros(n)
    vel_disp2_err = np.zeros(n)
    delta_chi2_line1_array = np.zeros(n)
    delta_chi2_line2_array = np.zeros(n)
    pure_redchi2_array = np.zeros(n)

    z_inds = [i for i, x in enumerate(z_abs) if not np.isnan(x) and x > 0]
    if verbose:
        logger.debug("performing final selection based on physical properties")
        logger.debug("only absorbers with conf_level > %s will be selected", conf_level)
    for m in z_inds:
        if len(fit_param[m]) > 0 and not np.all(np.isnan(fit_param[m])):
            z_new, z_new_error, fit_param_temp, fit_param_std_temp, EW_first_temp_mean, EW_second_temp_mean, EW_total_temp_mean, EW_first_error_temp, EW_second_error_temp, EW_total_error_temp, delta_chi2_line1, delta_chi2_line2 = measure_absorber_properties_double_gaussian(
                index=spec_index, wavelength=lam_obs, flux=residual, error=error,
                absorber_redshift=[z_abs[m]], bound=bound, use_kernel=absorber,
                d_pix=d_pix, use_covariance=use_covariance, nboot=nboot)
            z_new = float(z_new[0])
            z_new_error = float(z_new_error[0])
            delta_chi2_line1 = delta_chi2_line1[0]
            delta_chi2_line2 = delta_chi2_line2[0]

            if len(fit_param_temp[0]) > 0 and not np.all(np.isnan(fit_param_temp[0])):
                gaussian_parameters = np.array(fit_param_temp[0])
                gaussian_parameters_std = np.array(fit_param_std_temp[0])
                lam_rest = lam_obs / (1 + z_new)
                c0 = gaussian_parameters[1]
                c1 = gaussian_parameters[4]
                sig1, sig2 = gaussian_parameters[2], gaussian_parameters[5]
                sn1, sn2 = estimate_snr_for_lines(c0, c1, sig1, sig2, lam_rest, residual, error, logwave)
                disp_vel1, disp_vel2, disp_vel1_err, disp_vel2_err = vel_dispersion(c0, c1, gaussian_parameters[2],
                                        gaussian_parameters[5],
                                        gaussian_parameters_std[2], gaussian_parameters_std[5],
                                        resolution, z_new, lam_obs)

                # Use trapezoidal EWs if requested, otherwise fall back to Gaussian analytic EWs.
                # The same EW values are used consistently for doublet ratio, ew_snr cuts
                # (inside check_absorber_selection) and the stored catalog values.
                if trapz_ew_sigma is not None:
                    _tr = trapezoidal_ew(lam_obs, residual, error, z_new,
                                        c0, c1, sig1, sig2, n_sigma=trapz_ew_sigma)
                    ew1_val      = _tr['ew1']      if np.isfinite(_tr['ew1'])      else 0.0
                    ew2_val      = _tr['ew2']      if np.isfinite(_tr['ew2'])      else 0.0
                    ew_total_val = _tr['ew_total'] if np.isfinite(_tr['ew_total']) else 0.0
                    ew1_err_val  = _tr['ew1_err']  if np.isfinite(_tr['ew1_err'])  else 0.0
                    ew2_err_val  = _tr['ew2_err']  if np.isfinite(_tr['ew2_err'])  else 0.0
                    ew_total_err_val = _tr['ew_total_err'] if np.isfinite(_tr['ew_total_err']) else 0.0
                else:
                    ew1_val      = EW_first_temp_mean[0]
                    ew2_val      = EW_second_temp_mean[0]
                    ew_total_val = EW_total_temp_mean[0]
                    ew1_err_val  = EW_first_error_temp[0]
                    ew2_err_val  = EW_second_error_temp[0]
                    ew_total_err_val = EW_total_error_temp[0]

                if ew1_val > 0 and ew2_val > 0:
                    dr, dr_error = calculate_doublet_ratio(ew1_val, ew2_val,
                                                           ew1_err_val, ew2_err_val, f1, f2)
                    min_dr, max_dr = 1 - dr_error, line_ratio + dr_error
                    ew1_snr = ew1_val / ew1_err_val if ew1_err_val > 0 else 0.0
                    ew2_snr = ew2_val / ew2_err_val if ew2_err_val > 0 else 0.0
                else:
                    dr, min_dr, max_dr = 0, 0, -1
                    ew1_snr, ew2_snr = 0, 0
                good = check_absorber_selection(spec_index, z_new, gaussian_parameters, bound,
                                               lower_del_lam, c0, c1, upper_del_lam,
                                               sn1, sn_line1, sn2, sn_line2,
                                               disp_vel1, disp_vel2, min_dr, dr, max_dr,
                                               delta_chi2_line1, delta_chi2_line2,
                                               fit_param_std=fit_param_std_temp[0],
                                               conf_level=conf_level, vmax=_constants.MAX_VEL_DISPERSION, verbose=verbose)


                redchi2_doublet = fit_cost_double_gaussian(lam_obs,
                                                residual,
                                                error,
                                                z_new,
                                                gaussian_parameters,
                                                n_sigma_inner=2.5,
                                                min_pixels=8)

                if good:
                    pure_z_abs[m] = z_new
                    pure_gauss_fit[m] = fit_param_temp[0]
                    pure_gauss_fit_std[m] = fit_param_std_temp[0]
                    pure_ew_first_line_mean[m] = ew1_val
                    pure_ew_second_line_mean[m] = ew2_val
                    pure_ew_total_mean[m] = ew_total_val
                    pure_ew_first_line_error[m] = ew1_err_val
                    pure_ew_second_line_error[m] = ew2_err_val
                    pure_ew_total_error[m] = ew_total_err_val
                    redshift_err[m] = z_new_error
                    sn1_all[m] = sn1
                    sn2_all[m] = sn2
                    vel_disp1[m] = disp_vel1
                    vel_disp2[m] = disp_vel2
                    vel_disp1_err[m] = disp_vel1_err
                    vel_disp2_err[m] = disp_vel2_err
                    delta_chi2_line1_array[m] = delta_chi2_line1
                    delta_chi2_line2_array[m] = delta_chi2_line2
                    pure_redchi2_array[m] = redchi2_doublet

    return (pure_z_abs, pure_gauss_fit, pure_gauss_fit_std,
            pure_ew_first_line_mean, pure_ew_second_line_mean, pure_ew_total_mean,
            pure_ew_first_line_error, pure_ew_second_line_error, pure_ew_total_error,
                    redshift_err, sn1_all, sn2_all, vel_disp1, vel_disp2, vel_disp1_err, vel_disp2_err,
            delta_chi2_line1_array, delta_chi2_line2_array, pure_redchi2_array)


def _apply_false_positive_filters(pure_z_abs, sn1_all, sn2_all, lam_obs, residual, error,
                                   d_pix, absorber, logwave, pure_gauss_fit):
    # Remove absorbers that are likely contaminants.  For MgII we check whether
    # the feature is actually associated with a FeII system at a different redshift.
    # The third filter drops candidates that share contiguous pixels with a stronger
    # neighbour (i.e. they are probably sub-components of the same system).
    if absorber == 'MgII':
        match_abs1 = remove_Mg_falsely_come_from_Fe_absorber(pure_z_abs, lam_obs, residual,
                                                              error, d_pix, logwave)
    else:
        match_abs1 = -1 * np.ones(len(pure_z_abs))
    match_abs2 = z_abs_from_same_metal_absorber(pure_z_abs, lam_obs, residual, error,
                                                 d_pix, absorber, logwave)
    ind_z = contiguous_pixel_remover(pure_z_abs, sn1_all, sn2_all, absorber, pure_gauss_fit)
    return (match_abs1 == -1) & (match_abs2 == -1) & (ind_z == -1)


def convolution_method_absorber_finder_in_QSO_spectra(spec_index, absorber='MgII', lam_obs=None, residual=None, error=None, lam_search=None, unmsk_residual=None, ker_fwhm_pixels=5, coeff_sigma=2.5, mult_resi=1, d_pix=0.6, pm_pixel=200, sn_line1=3, sn_line2=2, use_covariance=False, logwave=True, verbose=True, nboot=None, conf_level=0.95, zabs_known=None, max_dv_known=None, trapz_ew_sigma=None, res_wave_start=None, res_val_start=None, res_wave_end=None, res_val_end=None, res_is_R=True):
    """
    Detect absorbers with doublet properties in SDSS quasar spectra using a
    convolution method. This function identifies potential absorbers based on
    user-defined threshold criteria, applies Gaussian fitting to reject false
    positives, and computes the equivalent widths (EWs) of the lines, returning
    the redshifts, EWs, and fitting parameters.

    Args:
        spec_index (int): Index of quasar in the spectra 2D array.
        absorber (str): Absorber name for searching doublets (MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, NaI). Default is 'MgII'.
        lam_obs (numpy.array): observed wavelength array.
        residual (numpy.array): residual (i.e. flux/continuum) array
        error (numpy.array): error on residuals
        lam_search (numpy.array): search observed wavelength array (i.e. region where absorber will be looked for).
        unmsk_residual (numpy.array): search residual array (residuals at search wavelength pixels)
        ker_fwhm_pixels (int or list): Kernel FWHM width(s) in pixels. Default is 5.
        coeff_sigma (float): Coefficient for sigma to apply threshold in the convolved array. Default is 2.5.
        mult_resi (float): Factor to shift the residual up or down. Default is 1.
        d_pix (float): Pixel distance for line separation during Gaussian fitting. Default is 0.6.
        pm_pixel (int): Pixel parameter for local noise estimation. Default is 200.
        sn_line1 (float): Signal-to-noise ratio threshold for line 1. Default is 3.
        sn_line2 (float): Signal-to-noise ratio threshold for line 2. Default is 2.
        use_covariance (bool): If True, use full covariance of scipy curve_fit for EW error calculation. Default is False.
        logwave (bool): If True, wavelength is on log scale (e.g. SDSS). Default is True.
        verbose (bool): If True, print detailed outputs for debugging. Default is True.
        nboot (int, optional): Number of bootstrap iterations for fitting. Default is None (disabled).
        conf_level (float): Minimum confidence level for chi2-based absorber selection. Default is 0.95.
        trapz_ew_sigma (float or None): If provided, equivalent widths are measured using the
            trapezoidal integration method with a window of ``+/- trapz_ew_sigma * sigma`` around
            each fitted line centre.  The same EW values are used for the doublet-ratio check,
            the ``ew_snr`` criterion inside ``check_absorber_selection``, and the stored catalog
            columns.  Gaussian fit parameters and their errors are always retained regardless of
            this setting.  Default is None (use Gaussian analytic EW).
        zabs_known (float or list, optional): Known absorber redshift(s) to validate. When given the
            convolution search is skipped entirely and the code goes straight to Gaussian fitting and
            selection for each supplied redshift.  Redshifts whose observed doublet falls outside the
            wavelength coverage of the spectrum are skipped with an info log message. Default is None.
        max_dv_known (float or None): Maximum allowed velocity offset (km/s) between the fitted
            redshift and the seed redshift when ``zabs_known`` is provided. Candidates whose
            fitted centre drifted further than this are rejected (``z_abs`` set to 0). When
            ``None`` (default), the value is read from ``_constants.ZABS_KNOWN_MAX_DV`` so it
            can be set once in the user constants file without touching call sites.
        res_wave_start (float, optional): Blue anchor wavelength (Angstrom) for the
            wavelength-dependent resolution model. When ``None`` (default), the
            pixel-spacing proxy is used instead.
        res_val_start (float, optional): Resolution value at ``res_wave_start``. Resolving
            power R if ``res_is_R=True``, or ``sigma_v`` in km/s otherwise. Default is
            ``None``.
        res_wave_end (float, optional): Red anchor wavelength (Angstrom) for the resolution
            model. Default is ``None``.
        res_val_end (float, optional): Resolution value at ``res_wave_end``. Same units as
            ``res_val_start``. Default is ``None``.
        res_is_R (bool, optional): If ``True`` (default), ``res_val_*`` are resolving powers
            R = lambda / FWHM_lambda. If ``False``, they are ``sigma_v`` in km/s.

    Returns:
        dict: Contains lists of various parameters related to detected absorbers.
            - index_spec (list): QSO spec index searched
            - z_abs (list): redshifts of absorbers detected
            - gauss_fit (list of arrays): gaussian fit parameters for each absorber
            - gauss_fit_std (list of arrays): errors on gaussian fit parameters for each absorber
            - ew_1_mean (list): Equivalent width of line 1 for each absorber
            - ew_2_mean (list): Equivalent width of line 2 for each absorber
            - ew_total_mean (list): Total Equivalent width of line 1 and line 2 for each absorber
            - ew_1_error (list): errors on Equivalent width of line 1 for each absorber
            - ew_2_error (list): errors on Equivalent width of line 2 for each absorber
            - ew_total_error (list): errors on Total Equivalent width of line 1 and line 2 for each absorber
            - z_abs_err (list): errors on redshifts of absorbers detected
            - sn_1 (list): SNR of line 1 for each absorber
            - sn_2 (list): SNR of line 2 for each absorber
            - vel_disp1 (list): rest-frame velocity dispersion of line 1 for each absorber (in km/s)
            - vel_disp2 (list): rest-frame velocity dispersion of line 2 for each absorber (in km/s)
            - delta_chi2 (list): min(delta_chi2_line1, delta_chi2_line2) for backward compatibility
            - delta_chi2_line1 (list): per-line delta_chi2 for line 1
            - delta_chi2_line2 (list): per-line delta_chi2 for line 2

    Note:
        ``z_abs`` in the returned dict uses two sentinel values when no absorber is detected:
        ``-1`` means the search **could not be attempted** (not enough wavelength pixels or the
        requested doublet falls outside the spectrum's wavelength coverage); ``0`` means the search
        **ran but found nothing** (convolution returned no candidates or all candidates failed
        Gaussian validation).
    """

    # zabs_known mode only requires lam_obs; the search window (lam_search) is not used.
    if lam_obs.size <= _constants.MIN_NPIXEL:
        if verbose:
            logger.info("Not enough wavelength pixels in spectrum, spec index = %s", spec_index)
        if zabs_known is not None:
            if isinstance(zabs_known, (int, float, np.floating, np.integer)):
                zabs_known = [float(zabs_known)]
            else:
                zabs_known = [float(z) for z in zabs_known]
            n = len(zabs_known)
            return _build_result(
                [spec_index] * n, [-1] * n, [[0, 0, 0, 0, 0, 0]] * n, [[0, 0, 0, 0, 0, 0]] * n,
                [0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n,
                [0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n,
                [0] * n, [] * n,
                zabs_known=zabs_known,
            )
        return _build_result(
            [spec_index], [-1], [[0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0]], [0], [0], [0],
            [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]
        )

    if zabs_known is None and (lam_search is None or lam_search.size <= _constants.MIN_NPIXEL):
        if verbose:
            logger.info("No wavelength pixels available in search region, spec index = %s", spec_index)
        return _build_result(
            [spec_index], [-1], [[0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0]], [0], [0], [0],
            [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]
        )

    line1, line2, f1, f2, line_ratio, line_sep, del_z = _get_doublet_constants(absorber)

    if verbose:
        logger.debug("For %s, theoretical oscillator strength ratio: %s", absorber, line_ratio)
        logger.debug("instrumental resolution will be calculated from wavelength array; wavelength pixels are assumed to be less than FWHM")

    # when zabs_known is given, lam_search may not be supplied; fall back to lam_obs for resolution
    lam_for_res = lam_obs if (lam_search is None or lam_search.size < 2) else lam_search
    wave_res, resolution, mean_resolution, del_sigma = _compute_resolution(
                                                            lam_for_res, lam_obs, line1, logwave,
                                                            res_wave_start=res_wave_start, res_val_start=res_val_start,
                                                            res_wave_end=res_wave_end, res_val_end=res_val_end,
                                                            res_is_R=res_is_R)

    if verbose:
        logger.debug("mean wave_resolution = %.5f, mean resolution per pixel = %.3f [km/s], del_sigma: %s", wave_res, mean_resolution, del_sigma)

    bound, lower_del_lam, upper_del_lam = _compute_fit_bounds(line1, line2, line_sep, d_pix, del_sigma)

    if zabs_known is not None:
        # direct validation path: skip the convolution search entirely
        if isinstance(zabs_known, (int, float, np.floating, np.integer)):
            zabs_known = [float(zabs_known)]
        else:
            zabs_known = [float(z) for z in zabs_known]
        lam_min, lam_max = lam_obs.min(), lam_obs.max()
        searchable = []
        out_of_range = []
        for z in zabs_known:
            obs1 = line1 * (1 + z)
            obs2 = line2 * (1 + z)
            if lam_min <= obs1 <= lam_max and lam_min <= obs2 <= lam_max:
                searchable.append(z)
            else:
                logger.info(
                    "spec index %s: %s at z=%.4f cannot be searched, observed doublet "
                    "[%.1f, %.1f] A is outside wavelength range [%.1f, %.1f] A",
                    spec_index, absorber, z, obs1, obs2, lam_min, lam_max)
                out_of_range.append(z)
        if not searchable:
            n = len(zabs_known)
            return _build_result(
                [spec_index] * n, [-1] * n, [[0, 0, 0, 0, 0, 0]] * n, [[0, 0, 0, 0, 0, 0]] * n,
                [0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n,
                [0] * n, [0] * n, [0] * n, [0] * n, [0] * n, [0] * n,
                [0] * n, [0] * n,
                zabs_known=zabs_known,
            )
        combined_final_our_z = searchable
        zabs_known_input = np.array(searchable)
    else:
        if isinstance(ker_fwhm_pixels, (int, float, np.integer, np.floating)):
            ker_fwhm_pixels = [ker_fwhm_pixels]
        ker_fwhm_pixels = np.asarray(ker_fwhm_pixels, dtype=float)
        ker_sigma_pixels = ker_fwhm_pixels / 2.355
        line_centre_weighted = (f1 * line1 + f2 * line2) / (f1 + f2)
        width_kernel = ker_sigma_pixels * wave_res if not logwave else line_centre_weighted * (10 ** (ker_sigma_pixels * wave_res) - 1.0)

        if verbose:
            logger.debug("kernel FWHM [pixels] = %s, kernel sigma [pixels] = %s, kernel sigma [Ang] = %s", ker_fwhm_pixels, ker_sigma_pixels, width_kernel)

        combined_final_our_z = _run_convolution_and_find_candidates(
            absorber, mult_resi, unmsk_residual, residual, lam_search, lam_obs, width_kernel,
            pm_pixel, coeff_sigma, line_ratio, line1, line2, del_z, logwave, wave_res, f1, f2,
            spec_index, verbose)
        if len(combined_final_our_z) == 0:
            return _build_result(
                [spec_index], [0], [[0, 0, 0, 0, 0, 0]], [[0, 0, 0, 0, 0, 0]], [0], [0], [0],
                [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]
            )
        zabs_known_input = None

    (pure_z_abs, pure_gauss_fit, pure_gauss_fit_std,
     pure_ew_first_line_mean, pure_ew_second_line_mean, pure_ew_total_mean,
     pure_ew_first_line_error, pure_ew_second_line_error, pure_ew_total_error,
      redshift_err, sn1_all, sn2_all, vel_disp1, vel_disp2, vel_disp1_err, vel_disp2_err,
     delta_chi2_line1_array, delta_chi2_line2_array, pure_redchi2_array) = _validate_candidates(
        spec_index, combined_final_our_z, lam_obs, residual, error, bound, absorber,
        d_pix, f1, f2, resolution, line_ratio, lower_del_lam, upper_del_lam,
        sn_line1, sn_line2, logwave, use_covariance, nboot, conf_level, verbose,
        trapz_ew_sigma=trapz_ew_sigma)

    if zabs_known_input is None:
        # convolution mode: discard failed candidates and remove false positives
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
        vel_disp1 = vel_disp1[valid_indices]
        vel_disp2 = vel_disp2[valid_indices]
        vel_disp1_err = vel_disp1_err[valid_indices]
        vel_disp2_err = vel_disp2_err[valid_indices]
        delta_chi2_line1_array = delta_chi2_line1_array[valid_indices]
        delta_chi2_line2_array = delta_chi2_line2_array[valid_indices]
        pure_redchi2_array = pure_redchi2_array[valid_indices]

        if verbose:
            logger.debug("final candidates: %s", pure_z_abs)

        if len(pure_z_abs) > 0:
            sel_indices = _apply_false_positive_filters(
                pure_z_abs, sn1_all, sn2_all, lam_obs, residual, error, d_pix, absorber, logwave, pure_gauss_fit)
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
            vel_disp1 = vel_disp1[sel_indices]
            vel_disp2 = vel_disp2[sel_indices]
            vel_disp1_err = vel_disp1_err[sel_indices]
            vel_disp2_err = vel_disp2_err[sel_indices]
            delta_chi2_line1_array = delta_chi2_line1_array[sel_indices]
            delta_chi2_line2_array = delta_chi2_line2_array[sel_indices]
            pure_redchi2_array = pure_redchi2_array[sel_indices]

        else:
            redshift_err = np.array([0])
            pure_z_abs = np.array([0])
            pure_gauss_fit = pure_gauss_fit_std = np.array([[0, 0, 0, 0, 0, 0]])
            pure_ew_first_line_mean = pure_ew_second_line_mean = pure_ew_total_mean = np.array([0])
            pure_ew_first_line_error = pure_ew_second_line_error = pure_ew_total_error = np.array([0])
            sn1_all = sn2_all = np.array([0])
            vel_disp1 = vel_disp2 = np.array([0])
            vel_disp1_err = vel_disp2_err = np.array([0])
            delta_chi2_line1_array = np.array([0])
            delta_chi2_line2_array = np.array([0])
            pure_redchi2_array = np.array([0])
    else:
        # known-z mode: keep all rows as-is (z_abs=0 for failed, fitted z for passed).
        # False-positive filters are not applied here since the redshifts were user-supplied.
        # Append any out-of-range entries at the end with z_abs=-1.

        # Apply max_dv_known cut: reject any detection whose fitted centre drifted too far
        # from the supplied seed redshift -- such systems are almost certainly a different feature.
        _max_dv = max_dv_known if max_dv_known is not None else _constants.ZABS_KNOWN_MAX_DV
        for m in range(len(pure_z_abs)):
            if pure_z_abs[m] > 0:
                dv = abs(pure_z_abs[m] - zabs_known_input[m]) / (1 + zabs_known_input[m]) * speed_of_light
                if dv > _max_dv:
                    if verbose:
                        logger.info(
                            "spec %s: rejecting %s at z_fit=%.4f (seed z=%.4f, dv=%.0f km/s > %.0f km/s limit)",
                            spec_index, absorber, pure_z_abs[m], zabs_known_input[m], dv, _max_dv)
                    pure_z_abs[m] = 0.0

        if verbose:
            logger.debug("final candidates: %s", pure_z_abs)
        if out_of_range:
            n_oor = len(out_of_range)
            pure_z_abs = np.concatenate([pure_z_abs, np.full(n_oor, -1.0)])
            pure_gauss_fit = np.concatenate([pure_gauss_fit, np.zeros((n_oor, 6))])
            pure_gauss_fit_std = np.concatenate([pure_gauss_fit_std, np.zeros((n_oor, 6))])
            pure_ew_first_line_mean = np.concatenate([pure_ew_first_line_mean, np.zeros(n_oor)])
            pure_ew_second_line_mean = np.concatenate([pure_ew_second_line_mean, np.zeros(n_oor)])
            pure_ew_total_mean = np.concatenate([pure_ew_total_mean, np.zeros(n_oor)])
            pure_ew_first_line_error = np.concatenate([pure_ew_first_line_error, np.zeros(n_oor)])
            pure_ew_second_line_error = np.concatenate([pure_ew_second_line_error, np.zeros(n_oor)])
            pure_ew_total_error = np.concatenate([pure_ew_total_error, np.zeros(n_oor)])
            redshift_err = np.concatenate([redshift_err, np.zeros(n_oor)])
            sn1_all = np.concatenate([sn1_all, np.zeros(n_oor)])
            sn2_all = np.concatenate([sn2_all, np.zeros(n_oor)])
            vel_disp1 = np.concatenate([vel_disp1, np.zeros(n_oor)])
            vel_disp2 = np.concatenate([vel_disp2, np.zeros(n_oor)])
            vel_disp1_err = np.concatenate([vel_disp1_err, np.zeros(n_oor)])
            vel_disp2_err = np.concatenate([vel_disp2_err, np.zeros(n_oor)])
            delta_chi2_line1_array = np.concatenate([delta_chi2_line1_array, np.zeros(n_oor)])
            delta_chi2_line2_array = np.concatenate([delta_chi2_line2_array, np.zeros(n_oor)])
            pure_redchi2_array = np.concatenate([pure_redchi2_array, np.zeros(n_oor)])
            zabs_known_input = np.concatenate([zabs_known_input, np.array(out_of_range)])

    not_found = max(1, len(pure_z_abs))
    index_spec = [spec_index for _ in range(not_found)]
    return _build_result(
        index_spec,
        pure_z_abs.tolist(),
        pure_gauss_fit.tolist(),
        pure_gauss_fit_std.tolist(),
        pure_ew_first_line_mean.tolist(),
        pure_ew_second_line_mean.tolist(),
        pure_ew_total_mean.tolist(),
        pure_ew_first_line_error.tolist(),
        pure_ew_second_line_error.tolist(),
        pure_ew_total_error.tolist(),
        redshift_err.tolist(),
        sn1_all.tolist(),
        sn2_all.tolist(),
        vel_disp1.tolist(),
        vel_disp2.tolist(),
        vel_disp1_err.tolist(),
        vel_disp2_err.tolist(),
        delta_chi2_line1_array.tolist(),
        delta_chi2_line2_array.tolist(),
        pure_redchi2_array.tolist(),
        zabs_known=zabs_known_input.tolist() if zabs_known_input is not None else None,
    )
