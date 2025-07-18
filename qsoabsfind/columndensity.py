"""
This script contains functions to calculate column densities for absorbers
using Apparent Optical Depth Method (AODM) of Savage & Sembach 1991

Paper link: https://ui.adsabs.harvard.edu/abs/1991ApJ...379..245S/abstract.
"""

import numpy as np
from astropy.table import Table, vstack
from multiprocessing import Pool
import time
from .absorberutils import calculate_doublet_ratio
from .config import load_constants


# We adopt the inverse-variance weighted column density for doublets with DR > 2 − DR_error. For systems with DR ≤ 2 − DR_error, we apply the Savage & Sembach (1991) correction to the weaker line when both transitions are measured. Otherwise, we adopt the weaker column density as a lower limit, or stronger line as a fallback if weaker line is unavailable.


# Constants
constants = load_constants()
speed_of_light = constants.speed_of_light
oscillator_strengths = constants.oscillator_parameters
lines = constants.lines
doublet_keys = constants.doublet_keys

def ss1991_correction(delta_logN):
    """
    Interpolate Savage & Sembach (1991) Table 4 to get Δlog N correction.

    Args:
        delta_logN (float): difference between logN of first and second lines

    Returns:
        correction based on Savage & Sembach 1991 paper (array)
    """
    delta_vals = np.array([
        0.000, 0.010, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070, 0.080, 0.090,
        0.100, 0.110, 0.110, 0.120, 0.130, 0.140, 0.150, 0.160, 0.170, 0.180,
        0.190, 0.200, 0.210, 0.220, 0.230, 0.240
    ])
    correction_vals = np.array([
        0.000, 0.010, 0.020, 0.030, 0.040, 0.051, 0.061, 0.073, 0.085, 0.097,
        0.111, 0.125, 0.125, 0.140, 0.157, 0.175, 0.195, 0.217, 0.243, 0.273,
        0.307, 0.348, 0.396, 0.453, 0.520, 0.600
    ])
    correction = np.interp(delta_logN, delta_vals, correction_vals, left=0.0, right=0.0)

    return correction


def optical_depth(F_lambda, sigma_F_lambda, continuum_error_frac):

    """ Function to calculate optical depth (tau) of absorption feature

    Args:
        F_lambda (array): continuum normalized flux
        sigma_F_lambda (array): errors on continuum normalized flux
        continuum_error_frac (float): assumed systematics on continuum normalized flux (default 5%)

    Returns:
        apparent optical depth array and corresponding error arrays
    """
    F_lambda = np.clip(F_lambda, 0.005, 1)  # Avoid log(0) issues
    tau = -np.log(F_lambda)
    sigma_tau_cont = np.log(1 + continuum_error_frac * np.exp(tau))
    sigma_F_lambda_inflated = np.sqrt(sigma_F_lambda**2 + sigma_tau_cont**2)
    sigma_tau = sigma_F_lambda_inflated / (F_lambda)
    return tau, sigma_tau

# Function to convert wavelength to velocity

def velocity_from_wavelength(lambda_array, lambda_0, z):

    """Function to convert wavelength into velocity pixels

    Args:
        lambda_array (array): observed wavelength
        lambda_0 (float): rest-frame wavelength of given absorber
        z (float): redshift of absorber

    Returns:
        velocities (observed and in rest-frame, array)

    """

    d_lambda = np.mean(lambda_array[1:] - lambda_array[:-1])
    lambda_obs  = lambda_0 * (1 + z)
    d_lambda = np.ones(lambda_array.size) * d_lambda
    dv = (lambda_array - lambda_obs) / lambda_obs * speed_of_light
    return speed_of_light * d_lambda / (lambda_obs), dv # Velocity in km/s

# Function to calculate total column density integrated over velocity range

def single_column_density(F_lambda, error, wavelength, z, f, lambda_0, continuum_error_frac, velocity_range):

    """Function to calculate apparent column density for one line using Savage & Sembach 1991 method

    Args:
        F_lambda (array): continuum normalized flux
        error (array): continuum normalized errors
        wavelength (array): observed wavelength
        z (float): redshift of absorber
        f (float): oscillator strength of line transition
        lambda_0 (float): rest-frame wavelength of given absorber
        continuum_error_frac (float): systematics on continuum normalized flux
        velocity_range (float): +/- velocity_range will be used for column density integration

    Returns:
        results (dict): dictionary containing column density and error

    """

    # Step 1: Convert wavelength to velocity

    v_array, dv_absorber = velocity_from_wavelength(wavelength, lambda_0, z)
    velocity_filter = (dv_absorber >= -velocity_range) & (dv_absorber <= velocity_range)
    F_lam = F_lambda[velocity_filter]
    err_F_lam = error[velocity_filter]
    sel = (~np.isnan(F_lam)) & (F_lam >0.005) & (F_lam < 1 + err_F_lam)
    F_lam = F_lam[sel]
    delta_dv_i = v_array[velocity_filter][sel]
    err_F_lam = err_F_lam[sel]
    tau, sigma_tau = optical_depth(F_lam, err_F_lam, continuum_error_frac)

    k_norm = 10**14.5762 / (lambda_0 * f)

    N_line = k_norm * np.nansum(tau* delta_dv_i)
    sig_N_line = k_norm * np.sqrt(np.nansum((sigma_tau * delta_dv_i) ** 2))

    if N_line > 0 and sig_N_line > 0 and not np.isnan(N_line):
        log_N = np.log10(N_line)
        err_log_N = sig_N_line / (N_line * np.log(10))
        flag = 1
    else:
        flag = -1
        N_line, sig_N_line = np.nan, np.nan
        log_N, err_log_N = np.nan, np.nan

    results = {'N':N_line, 'N_err':sig_N_line, 'logN':log_N, 'err_logN':err_log_N, 'flag':flag}

    return results


def total_column_density(F_lambda, error, wavelength, abs_cat, f1, f2, lambda1, lambda2, continuum_error_frac, velocity_range):

    """Function to calculate total apparent column density (inverse-variance weighted) for a doublet using Savage & Sembach 1991 method

    Args:
        F_lambda (array): continuum normalized flux
        error (array): continuum normalized errors
        wavelength (array): observed wavelength
        abs_cat (table): absorber table for the individual absorber (must contain Z_ABS, {absorber}_line12_EW, and {absorber}_line12_EW_ERROR, e.g. CIV_1548_EW)
        f1 (float): oscillator strength of first line
        f2 (float): oscillator strength of second line
        lambda1 (tuple): key and rest-frame wavelength of first line
        lambda2 (tuple): key and rest-frame wavelength of second line
        continuum_error_frac (float): systematics on continuum normalized flux
        velocity_range (float): +/- velocity_range will be used for column density integration

    Returns:
        results (dict): dictionary containing apparent column density and error
        and flag showing if its saturated

    Note:
        for fN flag, description:
        1: WEIGHTED, 2: FIRST, 3: SECOND, 4: Corrected weak line (partial saturation), 5: Lower limit from weak line (strong saturation), 6: Lower limit from strong (strong saturation and weak line is not available) -1: FAIL',

    """

    z = abs_cat["Z_ABS"]
    l1, l2 = lambda1[1], lambda2[1]
    ew1, ew2 = abs_cat[f"{lambda1[0].upper()}_EW"], abs_cat[f"{lambda2[0].upper()}_EW"]
    err_ew1, err_ew2 = abs_cat[f"{lambda1[0].upper()}_EW_ERROR"], abs_cat[f"{lambda2[0].upper()}_EW_ERROR"]

    dr, dr_error = calculate_doublet_ratio(ew1, ew2, err_ew1, err_ew2)
    sflag = 0 if dr > 2 - dr_error else 1

    results1 = single_column_density(F_lambda, error, wavelength, z, f1, l1,continuum_error_frac=continuum_error_frac, velocity_range=velocity_range)
    results2 = single_column_density(F_lambda, error, wavelength, z, f2, l2, continuum_error_frac=continuum_error_frac, velocity_range=velocity_range)

    N1, N2 = results1["N"], results2["N"]
    sig_N1, sig_N2 = results1["N_err"], results2["N_err"]
    flag1, flag2 = results1["flag"], results2["flag"]

    log_N, err_log_N = np.nan, np.nan

    if dr > 2 - dr_error:
        # Unsaturated
        if flag1 > 0 and flag2 > 0:
            w1, w2 = 1 / sig_N1**2, 1 / sig_N2**2
            N_tot = (w1 * N1 + w2 * N2) / (w1 + w2)
            N_tot_err = np.sqrt(1 / (w1 + w2))
            val_flag = 1  # weighted
        elif flag1 > 0:
            N_tot, N_tot_err = N1, sig_N1
            val_flag = 2  # first line only
        elif flag2 > 0:
            N_tot, N_tot_err = N2, sig_N2
            val_flag = 3  # second line only
        else:
            N_tot = N_tot_err = np.nan
            val_flag = -1
    else:
        # Saturated
        if flag1 > 0 and flag2 > 0:
            delta_logN = np.log10(N2) - np.log10(N1)
            correction = ss1991_correction(delta_logN)
            log_N = np.log10(N2) + correction
            err_log_N = sig_N2 / (N2 * np.log(10))
            N_tot = 10**log_N
            N_tot_err = N_tot * err_log_N * np.log(10)
            val_flag = 4  # S&S corrected weak line
        elif flag2 > 0:
            N_tot, N_tot_err = N2, sig_N2
            val_flag = 5  # lower limit from weak line
        elif flag1 > 0:
            N_tot, N_tot_err = N1, sig_N1
            val_flag = 6  # lower limit from strong line
        else:
            N_tot = N_tot_err = np.nan
            val_flag = -1

    if np.isnan(log_N):
        log_N = np.log10(N_tot) if np.isfinite(N_tot) and N_tot > 0 else np.nan
        err_log_N = N_tot_err / (N_tot * np.log(10)) if np.isfinite(N_tot) and N_tot > 0 else np.nan

    results = Table({'LOG10N': [log_N], 'SIG_LOG10N': [err_log_N], 'SATURATION': [sflag], 'fN': [val_flag]})

    return results

def compute_single_column_density(args):
    """Function to compute column density of one absorbers
    """
    flux, error, wavelength, tt_row, f1, f2, l1, l2, dv = args
    return total_column_density(flux, error, wavelength, tt_row, f1, f2, l1, l2, velocity_range=dv)


def return_total_column_density_table(input, absorber, output, continuum_error_frac=0.05, dv=300, nproc=None):

    """ Function to calculate total column density of metal doublets using
    apparent optical depth method

    Args:
        input (str): input spectra file
        absorber (str): absorber name (MgII, CIV, OVI, FeII, AlIII, SiIV, NV)
        output (str): output absorber catalog filename
        continuum_error_frac (float): systematics on continuum normalized flux (default: 0.05)
        dv (float): maximum velocity range to be considered for optical depth calculation (default 300 km/s)
        nproc (int): number of cpus for multiprocessing

    Returns:
        appends an extra HDU: COLUMN_DENSITY in the absorber catalog output filename
    """

    from .spec import QSOSpecRead
    start = time.time()

    tt = Table.read(output, hdu="ABSORBER")
    spectra = QSOSpecRead(input, autoload=True, index=tt["INDEX_SPEC"])
    F_lambda = spectra.flux
    error_F_lambda = spectra.error
    wavelength = spectra.wavelength

    f1, f2 = oscillator_strengths[absorber + '_f1'], oscillator_strengths[absorber + '_f2']
    l1_key, l2_key = doublet_keys[absorber][0], doublet_keys[absorber][1]
    l1, l2 = (l1_key, lines[l1_key]), (l2_key, lines[l2_key])

    nabs = len(tt)

    args_list = [
        (F_lambda[i], error_F_lambda[i], wavelength, tt[i], f1, f2, l1, l2, continuum_error_frac, dv)
        for i in range(nabs)
    ]

    print(f"INFO: Starting column density calculation with {nproc} processes")

    with Pool(nproc) as pool:
        results = pool.map(compute_single_column_density, args_list)

    # Combine into single table
    N_table = vstack(results)
    for col in N_table.colnames:
        if "10N" in col:
            N_table[col].unit = "cm^-2"
            N_table[col] = N_table[col].astype('float64')
        else:
            N_table[col] = N_table[col].astype('int32')

    print(f'INFO: Num of zeros = {np.sum(N_table["LOG10N"].data==0)}')
    print(f'INFO: Column density took = {time.time()-start:.3f} [sec]')

    return N_table
