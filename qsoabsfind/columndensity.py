"""
Column-density measurements for absorption-line doublets using the
Apparent Optical Depth Method (AODM; Savage & Sembach 1991).

Important conventions
---------------------
* ``velocity_range`` is a HALF-WIDTH: the integration interval is
  [-velocity_range, +velocity_range] km/s around each transition.
* Pixels at or below ``AODM_FLUX_CLIP_MIN`` are retained at the floor and
  flagged as saturated/lower-limit pixels; they are never discarded.
* Flux values above unity are NOT clipped. Their negative apparent optical
  depth is retained so noise is not treated asymmetrically.
* Statistical and continuum-placement uncertainties are propagated separately.
  Continuum placement is treated as a correlated multiplicative systematic.
* Unresolved saturation is diagnosed from the difference between the
  velocity-integrated AOD columns of the weak and strong transitions.

Flags:

Total doublet AODM column.

    SATURATION
    ----------
     0 : no significant AOD evidence for unresolved saturation
     1 : unresolved saturation; S&S correction applied
     2 : severe saturation / outside S&S calibration; lower limit
     3 : saturation indeterminate because only one line is usable
    -2 : inconsistent doublet (N_weak significantly < N_strong)
    -1 : failed

    fN
    --
     1 : inverse-variance weighted doublet
     2 : line 1 only
     3 : line 2 only
     4 : S&S-corrected weak line
     5 : lower limit from weak line
     6 : lower limit from strong line
     7 : inconsistent doublet
    -1 : failed
"""

import time
from multiprocessing import Pool
import logging
import numpy as np
from astropy.table import Table, vstack

logger = logging.getLogger(__name__)

from .constants import lines, oscillator_parameters, speed_of_light, doublet_keys
from . import constants as _constants


_SS_DELTA_LOGN = np.arange(0.00, 0.25, 0.01)

_SS_CORRECTION = np.array([
    0.000, 0.010, 0.020, 0.030, 0.040,
    0.051, 0.061, 0.073, 0.085, 0.097,
    0.111, 0.125, 0.140, 0.157, 0.175,
    0.195, 0.217, 0.243, 0.273, 0.307,
    0.348, 0.396, 0.453, 0.520, 0.600,
], dtype=float)

_AODM_NORM = 3.768e14


def ss1991_correction(delta_logN, return_slope=False):
    """Interpolate Savage & Sembach (1991) Table 4."""
    if not np.isfinite(delta_logN) or delta_logN < 0.0 or delta_logN > 0.24:
        return (np.nan, np.nan) if return_slope else np.nan

    correction = float(np.interp(delta_logN, _SS_DELTA_LOGN, _SS_CORRECTION))

    if not return_slope:
        return correction

    if delta_logN >= _SS_DELTA_LOGN[-1]:
        i = len(_SS_DELTA_LOGN) - 2
    else:
        i = np.searchsorted(_SS_DELTA_LOGN, delta_logN, side="right") - 1
        i = int(np.clip(i, 0, len(_SS_DELTA_LOGN) - 2))

    slope = (
        (_SS_CORRECTION[i + 1] - _SS_CORRECTION[i])
        / (_SS_DELTA_LOGN[i + 1] - _SS_DELTA_LOGN[i])
    )
    return correction, float(slope)


def optical_depth(F_lambda, sigma_F_lambda, continuum_error_frac=0.0):
    """
    Apparent optical depth and statistical uncertainty.

    continuum_error_frac is kept for API compatibility. Continuum-placement
    uncertainty is handled coherently in single_column_density().
    """
    F_lambda = np.asarray(F_lambda, dtype=float)
    sigma_F_lambda = np.asarray(sigma_F_lambda, dtype=float)

    floor = float(_constants.AODM_FLUX_CLIP_MIN)
    flux_use = np.maximum(F_lambda, floor)

    with np.errstate(divide="ignore", invalid="ignore"):
        tau = -np.log(flux_use)
        sigma_tau = sigma_F_lambda / flux_use

    return tau, sigma_tau


def velocity_from_wavelength(lambda_array, lambda_0, z, logwave=False):
    """
    Convert observed wavelength to velocity relative to lambda_0*(1+z).

    Returns per-pixel velocity width and velocity coordinate, both in km/s.
    """
    lambda_array = np.asarray(lambda_array, dtype=float)

    if lambda_array.size < 2:
        nan = np.full(lambda_array.size, np.nan)
        return nan.copy(), nan.copy()

    lambda_obs = float(lambda_0) * (1.0 + float(z))
    velocity = speed_of_light * (lambda_array / lambda_obs - 1.0)
    dv_pixel = np.abs(np.gradient(velocity))

    return dv_pixel, velocity


def _empty_single_result():
    return {
        "N": np.nan,
        "N_err": np.nan,
        "N_err_stat": np.nan,
        "N_err_cont": np.nan,
        "N_cont_plus": np.nan,
        "N_cont_minus": np.nan,
        "logN": np.nan,
        "err_logN": np.nan,
        "flag": -1,
        "is_lower_limit": False,
        "n_saturated": 0,
        "n_pixels": 0,
    }


def _integrated_aod_column(flux, dv_pixel, f, lambda_0):
    floor = float(_constants.AODM_FLUX_CLIP_MIN)
    flux_use = np.maximum(np.asarray(flux, dtype=float), floor)
    tau = -np.log(flux_use)

    k_norm = _AODM_NORM / (float(lambda_0) * float(f))
    return k_norm * np.sum(tau * dv_pixel)


def single_column_density(
    F_lambda,
    error,
    wavelength,
    z,
    f,
    lambda_0,
    continuum_error_frac,
    velocity_range,
    logwave,
):
    """
    Velocity-integrated AODM column for one transition.

    velocity_range is the HALF-WIDTH in km/s.
    """
    F_lambda = np.asarray(F_lambda, dtype=float)
    error = np.asarray(error, dtype=float)
    wavelength = np.asarray(wavelength, dtype=float)

    if not (
        F_lambda.shape == error.shape == wavelength.shape
        and F_lambda.ndim == 1
        and wavelength.size >= 2
        and np.isfinite(z)
        and np.isfinite(f)
        and f > 0
        and np.isfinite(lambda_0)
        and lambda_0 > 0
        and np.isfinite(velocity_range)
        and velocity_range > 0
    ):
        return _empty_single_result()

    dv_pixel, velocity = velocity_from_wavelength(
        wavelength, lambda_0, z, logwave=logwave
    )

    in_window = (velocity >= -velocity_range) & (velocity <= velocity_range)
    good = (
        in_window
        & np.isfinite(F_lambda)
        & np.isfinite(error)
        & np.isfinite(dv_pixel)
        & (error > 0)
        & (dv_pixel > 0)
    )

    if np.count_nonzero(good) < 2:
        return _empty_single_result()

    flux = F_lambda[good]
    err = error[good]
    dv = dv_pixel[good]

    floor = float(_constants.AODM_FLUX_CLIP_MIN)
    saturated = flux <= floor
    flux_use = np.maximum(flux, floor)

    tau, sigma_tau_stat = optical_depth(flux_use, err, continuum_error_frac=0.0)

    k_norm = _AODM_NORM / (float(lambda_0) * float(f))
    N_line = k_norm * np.sum(tau * dv)
    sig_N_stat = k_norm * np.sqrt(np.sum((sigma_tau_stat * dv) ** 2))

    eps = 0.0 if continuum_error_frac is None else float(continuum_error_frac)
    if not np.isfinite(eps) or eps < 0 or eps >= 1:
        raise ValueError("continuum_error_frac must satisfy 0 <= value < 1")

    if eps > 0:
        flux_cont_plus = flux / (1.0 + eps)
        flux_cont_minus = flux / (1.0 - eps)

        N_cont_plus = _integrated_aod_column(flux_cont_plus, dv, f, lambda_0)
        N_cont_minus = _integrated_aod_column(flux_cont_minus, dv, f, lambda_0)

        sig_N_cont = max(
            abs(N_cont_plus - N_line),
            abs(N_cont_minus - N_line),
        )
    else:
        N_cont_plus = N_line
        N_cont_minus = N_line
        sig_N_cont = 0.0

    sig_N_line = np.sqrt(sig_N_stat**2 + sig_N_cont**2)

    if (
        np.isfinite(N_line)
        and N_line > 0
        and np.isfinite(sig_N_line)
        and sig_N_line >= 0
    ):
        log_N = np.log10(N_line)
        err_log_N = sig_N_line / (N_line * np.log(10.0))
        flag = 1
    else:
        result = _empty_single_result()
        result["is_lower_limit"] = bool(np.any(saturated))
        result["n_saturated"] = int(np.sum(saturated))
        result["n_pixels"] = int(flux.size)
        return result

    return {
        "N": float(N_line),
        "N_err": float(sig_N_line),
        "N_err_stat": float(sig_N_stat),
        "N_err_cont": float(sig_N_cont),
        "N_cont_plus": float(N_cont_plus),
        "N_cont_minus": float(N_cont_minus),
        "logN": float(log_N),
        "err_logN": float(err_log_N),
        "flag": flag,
        "is_lower_limit": bool(np.any(saturated)),
        "n_saturated": int(np.sum(saturated)),
        "n_pixels": int(flux.size),
    }


def _combine_stat_and_cont_error(N, N_err_stat, N_cont_plus, N_cont_minus):
    if not np.isfinite(N) or N <= 0 or not np.isfinite(N_err_stat):
        return np.nan, np.nan

    cont_terms = []
    if np.isfinite(N_cont_plus):
        cont_terms.append(abs(N_cont_plus - N))
    if np.isfinite(N_cont_minus):
        cont_terms.append(abs(N_cont_minus - N))

    N_err_cont = max(cont_terms) if cont_terms else 0.0
    N_err = np.sqrt(N_err_stat**2 + N_err_cont**2)
    return float(N_err), float(N_err_cont)


def _single_line_output(result, line_number, is_weak):
    if result["flag"] <= 0:
        return np.nan, np.nan, -1, -1, 0

    N = result["N"]
    N_err = result["N_err"]

    if result["is_lower_limit"]:
        fN = 5 if is_weak else 6
        saturation = 2
        lower_limit = 1
    else:
        fN = 2 if line_number == 1 else 3
        saturation = 3
        lower_limit = 0

    return N, N_err, saturation, fN, lower_limit


def total_column_density(
    F_lambda,
    error,
    wavelength,
    abs_cat,
    f1,
    f2,
    lambda1,
    lambda2,
    continuum_error_frac,
    velocity_range,
    logwave,
):
    """
    Total doublet AODM column.

    SATURATION
    ----------
     0 : no significant AOD evidence for unresolved saturation
     1 : unresolved saturation; S&S correction applied
     2 : severe saturation / outside S&S calibration; lower limit
     3 : saturation indeterminate because only one line is usable
    -2 : inconsistent doublet (N_weak significantly < N_strong)
    -1 : failed

    fN
    --
     1 : inverse-variance weighted doublet
     2 : line 1 only
     3 : line 2 only
     4 : S&S-corrected weak line
     5 : lower limit from weak line
     6 : lower limit from strong line
     7 : inconsistent doublet
    -1 : failed
    """
    z = float(abs_cat["Z_ABS"])
    l1, l2 = float(lambda1[1]), float(lambda2[1])

    results1 = single_column_density(
        F_lambda, error, wavelength, z, f1, l1,
        continuum_error_frac=continuum_error_frac,
        velocity_range=velocity_range,
        logwave=logwave,
    )
    results2 = single_column_density(
        F_lambda, error, wavelength, z, f2, l2,
        continuum_error_frac=continuum_error_frac,
        velocity_range=velocity_range,
        logwave=logwave,
    )

    strength1 = float(f1) * l1
    strength2 = float(f2) * l2

    if strength1 >= strength2:
        strong, weak = results1, results2
        strong_line_number, weak_line_number = 1, 2
    else:
        strong, weak = results2, results1
        strong_line_number, weak_line_number = 2, 1

    valid1 = results1["flag"] > 0
    valid2 = results2["flag"] > 0

    delta_logN = np.nan
    sig_delta_logN = np.nan
    lower_limit = 0

    if not valid1 and not valid2:
        N_tot = N_tot_err = np.nan
        saturation = -1
        val_flag = -1

    elif valid1 and not valid2:
        is_weak = weak_line_number == 1
        N_tot, N_tot_err, saturation, val_flag, lower_limit = _single_line_output(
            results1, line_number=1, is_weak=is_weak
        )

    elif valid2 and not valid1:
        is_weak = weak_line_number == 2
        N_tot, N_tot_err, saturation, val_flag, lower_limit = _single_line_output(
            results2, line_number=2, is_weak=is_weak
        )

    else:
        Ns, Nw = strong["N"], weak["N"]
        sigNs_stat = strong["N_err_stat"]
        sigNw_stat = weak["N_err_stat"]

        logNs = np.log10(Ns)
        logNw = np.log10(Nw)

        sig_logNs_stat = sigNs_stat / (Ns * np.log(10.0))
        sig_logNw_stat = sigNw_stat / (Nw * np.log(10.0))

        delta_logN = logNw - logNs
        sig_delta_logN = np.sqrt(sig_logNw_stat**2 + sig_logNs_stat**2)

        if weak["is_lower_limit"] or strong["is_lower_limit"]:
            N_tot = Nw
            N_tot_err = weak["N_err"]
            saturation = 2
            val_flag = 5
            lower_limit = 1

        elif delta_logN < -_constants.AODM_INCONSISTENT_SIGMA * sig_delta_logN:
            N_tot = N_tot_err = np.nan
            saturation = -2
            val_flag = 7

        elif delta_logN <= sig_delta_logN:
            if sigNs_stat > 0 and sigNw_stat > 0:
                ws = 1.0 / sigNs_stat**2
                ww = 1.0 / sigNw_stat**2

                N_tot = (ws * Ns + ww * Nw) / (ws + ww)
                N_err_stat = np.sqrt(1.0 / (ws + ww))

                N_plus = (
                    ws * strong["N_cont_plus"] + ww * weak["N_cont_plus"]
                ) / (ws + ww)
                N_minus = (
                    ws * strong["N_cont_minus"] + ww * weak["N_cont_minus"]
                ) / (ws + ww)

                N_tot_err, _ = _combine_stat_and_cont_error(
                    N_tot, N_err_stat, N_plus, N_minus
                )

                saturation = 0
                val_flag = 1
            else:
                N_tot = N_tot_err = np.nan
                saturation = -1
                val_flag = -1

        elif delta_logN <= 0.24:
            correction, slope = ss1991_correction(
                delta_logN, return_slope=True
            )

            if np.isfinite(correction) and np.isfinite(slope):
                logN_corr = logNw + correction
                N_tot = 10.0**logN_corr

                sig_logN_stat = np.sqrt(
                    (1.0 + slope)**2 * sig_logNw_stat**2
                    + slope**2 * sig_logNs_stat**2
                )
                N_err_stat = N_tot * np.log(10.0) * sig_logN_stat

                def _shifted_corrected_N(Nw_shift, Ns_shift):
                    if (
                        not np.isfinite(Nw_shift)
                        or not np.isfinite(Ns_shift)
                        or Nw_shift <= 0
                        or Ns_shift <= 0
                    ):
                        return np.nan

                    dlogw = np.log10(Nw_shift) - logNw
                    dlogs = np.log10(Ns_shift) - logNs
                    dlogcorr = (1.0 + slope) * dlogw - slope * dlogs

                    return 10.0**(logN_corr + dlogcorr)

                N_plus = _shifted_corrected_N(
                    weak["N_cont_plus"], strong["N_cont_plus"]
                )
                N_minus = _shifted_corrected_N(
                    weak["N_cont_minus"], strong["N_cont_minus"]
                )

                N_tot_err, _ = _combine_stat_and_cont_error(
                    N_tot, N_err_stat, N_plus, N_minus
                )

                saturation = 1
                val_flag = 4
            else:
                N_tot = Nw
                N_tot_err = weak["N_err"]
                saturation = 2
                val_flag = 5
                lower_limit = 1

        else:
            N_tot = Nw
            N_tot_err = weak["N_err"]
            saturation = 2
            val_flag = 5
            lower_limit = 1

    if np.isfinite(N_tot) and N_tot > 0:
        log_N = np.log10(N_tot)
        err_log_N = (
            N_tot_err / (N_tot * np.log(10.0))
            if np.isfinite(N_tot_err) and N_tot_err >= 0
            else np.nan
        )
    else:
        log_N = np.nan
        err_log_N = np.nan

    return Table({
        "LOG10N": [log_N],
        "SIG_LOG10N": [err_log_N],
        "SATURATION": [int(saturation)],
        "fN": [int(val_flag)],
        "LOWER_LIMIT": [int(lower_limit)],
        "DELTA_LOGN": [delta_logN],
        "SIG_DELTA_LOGN": [sig_delta_logN],
        "NPIX_SAT_STRONG": [
            int(strong["n_saturated"]) if valid1 and valid2 else -1
        ],
        "NPIX_SAT_WEAK": [
            int(weak["n_saturated"]) if valid1 and valid2 else -1
        ],
    })


def compute_single_column_density(args):
    """Multiprocessing wrapper for one absorber."""
    (
        flux, error, wavelength, tt_row,
        f1, f2, l1, l2,
        continuum_error_frac, dv, logwave,
    ) = args

    return total_column_density(
        flux, error, wavelength, tt_row, f1, f2, l1, l2,
        continuum_error_frac=continuum_error_frac,
        velocity_range=dv,
        logwave=logwave,
    )


def _sentinel_column_density_table():
    return Table({
        "LOG10N": [np.nan],
        "SIG_LOG10N": [np.nan],
        "SATURATION": [-1],
        "fN": [-1],
        "LOWER_LIMIT": [-1],
        "DELTA_LOGN": [np.nan],
        "SIG_DELTA_LOGN": [np.nan],
        "NPIX_SAT_STRONG": [-1],
        "NPIX_SAT_WEAK": [-1],
    })


def return_total_column_density_table(
    spectra_fits,
    absorber,
    output,
    continuum_error_frac=0.05,
    dv=300,
    logwave=False,
    nproc=None,
):
    """
    Calculate AODM column densities for all absorbers.

    dv is the HALF-WIDTH of the integration interval in km/s.
    """
    from .datamodel import QSOSpecRead

    start = time.time()

    tt = Table.read(output, hdu="ABSORBER")
    spectra = QSOSpecRead(
        spectra_fits,
        autoload=True,
        index=tt["INDEX_SPEC"],
    )

    F_lambda = np.asarray(spectra.flux)
    error_F_lambda = np.asarray(spectra.error)
    wavelength = np.asarray(spectra.wavelength)

    f1 = oscillator_parameters[absorber + "_f1"]
    f2 = oscillator_parameters[absorber + "_f2"]

    l1_key, l2_key = doublet_keys[absorber][0], doublet_keys[absorber][1]
    l1 = (l1_key, lines[l1_key])
    l2 = (l2_key, lines[l2_key])

    mean_lambda = 0.5 * (l1[1] + l2[1])
    doublet_sep_kms = (
        speed_of_light * abs(l2[1] - l1[1]) / mean_lambda
    )

    if 2.0 * dv >= doublet_sep_kms:
        logger.warning(
            "AODM integration windows overlap for %s: each line uses +/- %.1f km/s, "
            "while the doublet separation is %.1f km/s. For NaI, +/-150 km/s "
            "is safer than +/-300 km/s.",
            absorber,
            dv,
            doublet_sep_kms,
        )

    nabs = len(tt)
    valid_mask = np.asarray(tt["Z_ABS"], dtype=float) > 0
    sentinel = _sentinel_column_density_table()

    def _wave_for_row(i):
        return wavelength[i] if wavelength.ndim == 2 else wavelength

    args_list = [
        (
            F_lambda[i],
            error_F_lambda[i],
            _wave_for_row(i),
            tt[i],
            f1,
            f2,
            l1,
            l2,
            continuum_error_frac,
            dv,
            logwave,
        )
        for i in range(nabs)
        if valid_mask[i]
    ]

    logger.info(
        "Starting column density calculation with %s processes",
        nproc,
    )

    if args_list:
        with Pool(nproc) as pool:
            valid_results = pool.map(
                compute_single_column_density,
                args_list,
            )
    else:
        valid_results = []

    valid_iter = iter(valid_results)
    ordered = [
        next(valid_iter) if valid else sentinel
        for valid in valid_mask
    ]

    N_table = (
        vstack(ordered)
        if ordered
        else _sentinel_column_density_table()[:0]
    )

    for col in [
        "LOG10N",
        "SIG_LOG10N",
        "DELTA_LOGN",
        "SIG_DELTA_LOGN",
    ]:
        N_table[col] = N_table[col].astype("float64")

    for col in [
        "SATURATION",
        "fN",
        "LOWER_LIMIT",
        "NPIX_SAT_STRONG",
        "NPIX_SAT_WEAK",
    ]:
        N_table[col] = N_table[col].astype("int32")

    N_table["LOG10N"].description = "log10[N/(cm^-2)]"
    N_table["SIG_LOG10N"].description = "1-sigma uncertainty in log10 N (dex)"
    N_table["DELTA_LOGN"].description = (
        "log10(N_weak) - log10(N_strong) from AODM"
    )
    N_table["SIG_DELTA_LOGN"].description = (
        "Statistical 1-sigma uncertainty on DELTA_LOGN (dex)"
    )

    logger.info(
        "Column density calculation finished in %.3f sec for %d absorber rows",
        time.time() - start,
        nabs,
    )

    return N_table
