#!/usr/bin/env python

import numpy as np
import argparse
import time
from multiprocessing import Pool
from astropy.io import fits
from importlib import import_module
from .absfinder import convolution_method_absorber_finder_in_QSO_spectra
from .io import save_results_to_fits
import re

def run_convolution_method_absorber_finder_QSO_spectra(fits_file, spec_index, absorber, ker_width_pix, coeff_sigma, mult_resi, d_pix, pm_pixel, sn_line1, sn_line2, use_covariance):
    return convolution_method_absorber_finder_in_QSO_spectra(fits_file, spec_index, absorber, ker_width_pix, coeff_sigma, mult_resi, d_pix, pm_pixel, sn_line1, sn_line2, use_covariance)


def parse_qso_sequence(qso_sequence):
    """
    Parse a bash-like sequence or a single integer to generate QSO indices.

    Parameters:
    qso_sequence (str or int): Bash-like sequence (e.g., '1-1000', '1-1000:10') or an integer.

    Returns:
    numpy.array: Array of QSO indices.
    """
    if isinstance(qso_sequence, int):
        return np.arange(qso_sequence)

    match = re.match(r"(\d+)-(\d+)(?::(\d+))?", qso_sequence)
    if match:
        start, end, step = match.groups()
        start, end = int(start), int(end)
        step = int(step) if step else 1
        return np.arange(start, end + 1, step)

    raise ValueError("Invalid QSO sequence format. Use 'start-end[:step]' or an integer.")

def parallel_convolution_method_absorber_finder_QSO_spectra(fits_file, spec_indices, absorber='MgII', ker_width_pix=[3, 4, 5, 6, 7, 8], coeff_sigma=2.5, mult_resi=1, d_pix=0.6, pm_pixel=200, sn_line1=3, sn_line2=2, use_covariance=False, n_jobs=1):
    """
    Run convolution_method_absorber_finder_in_QSO_spectra in parallel using multiprocessing.

    Parameters:
    fits_file (str): Path to the FITS file containing Normalized QSO spectra.
    spec_indices (list or numpy.array): Indices of quasars in the data matrix.
    absorber (str): Absorber name for searching doublets (MgII, CIV). Default is 'MgII'.
    ker_width_pix (list): List of kernel widths in pixels. Default is [3, 4, 5, 6, 7, 8].
    coeff_sigma (float): Coefficient for sigma to apply threshold in the convolved array. Default is 2.5.
    mult_resi (float): Factor to shift the residual up or down. Default is 1.
    d_pix (float): Pixel distance for line separation during Gaussian fitting. Default is 0.6.
    pm_pixel (int): Pixel parameter for local noise estimation (default 200).
    sn_line1 (float): Signal-to-noise ratio for thresholding for line1.
    sn_line2 (float): Signal-to-noise ratio for thresholding for line2.
    use_covariance (bool): If want to use full covariance of scipy curve_fit for EW error calculation (default is False).
    n_jobs (int): Number of parallel jobs to run.

    Returns:
    dict: A dictionary containing combined results from all parallel runs.
    """
    params_list = [(fits_file, spec_index, absorber, ker_width_pix, coeff_sigma, mult_resi, d_pix, pm_pixel, sn_line1, sn_line2, use_covariance) for spec_index in spec_indices]

    print(params_list)
    # Run the jobs in parallel
    with Pool(processes=n_jobs) as pool:
        results = pool.starmap(run_convolution_method_absorber_finder_QSO_spectra, params_list)

    # Combine the results
    combined_results = {
        'index_spec': [],
        'z_abs': [],
        'gauss_fit': [],
        'gauss_fit_std': [],
        'ew_1_mean': [],
        'ew_2_mean': [],
        'ew_total_mean': [],
        'ew_1_error': [],
        'ew_2_error': [],
        'ew_total_error': [],
        'z_abs_err': [],
        'sn_1': [],
        'sn_2': [],
    }

    for result in results:
        (index_spec, z_abs, gauss_fit, gauss_fit_std, ew_1_mean, ew_2_mean, ew_total_mean,
         ew_1_error, ew_2_error, ew_total_error, z_abs_err, sn_1, sn_2) = result

        valid_indices = np.array(z_abs) > 0

        combined_results['index_spec'].extend(np.array(index_spec)[valid_indices])
        combined_results['z_abs'].extend(np.array(z_abs)[valid_indices])
        combined_results['gauss_fit'].extend(np.array(gauss_fit)[valid_indices])
        combined_results['gauss_fit_std'].extend(np.array(gauss_fit_std)[valid_indices])
        combined_results['ew_1_mean'].extend(np.array(ew_1_mean)[valid_indices])
        combined_results['ew_2_mean'].extend(np.array(ew_2_mean)[valid_indices])
        combined_results['ew_total_mean'].extend(np.array(ew_total_mean)[valid_indices])
        combined_results['ew_1_error'].extend(np.array(ew_1_error)[valid_indices])
        combined_results['ew_2_error'].extend(np.array(ew_2_error)[valid_indices])
        combined_results['ew_total_error'].extend(np.array(ew_total_error)[valid_indices])
        combined_results['z_abs_err'].extend(np.array(z_abs_err)[valid_indices])
        combined_results['sn_1'].extend(np.array(sn_1)[valid_indices])
        combined_results['sn_2'].extend(np.array(sn_2)[valid_indices])

    return combined_results

def main():
    parser = argparse.ArgumentParser(description='Run convolution method for QSO spectra in parallel.')
    parser.add_argument('--input-fits-file', type=str, required=True, help='Path to the input FITS file.')
    parser.add_argument('--n-qso', required=True, help="Number of QSO spectra to process, or a bash-like sequence (e.g., '1-1000', '1-1000:10').")
    parser.add_argument('--absorber', type=str, required=True, help='Absorber name for searching doublets (MgII, CIV).')
    parser.add_argument('--constant-file', type=str, required=True, help='Path to the constants .py file.')
    parser.add_argument('--output', type=str, required=True, help='Path to the output FITS file.')
    parser.add_argument('--headers', type=str, nargs='+', help='Headers for the output FITS file in the format NAME=VALUE.')
    parser.add_argument('--n-tasks', type=int, required=True, help='Number of tasks.')
    parser.add_argument('--ncpus', type=int, required=True, help='Number of CPUs per task.')

    args = parser.parse_args()

    # Import constants from the provided file
    constants = import_module(args.constant_file.rstrip('.py'))

    # Prepare headers
    headers = {}
    if args.headers:
        for header in args.headers:
            key, value = header.split('=')
            headers[key] = value

    # Start timing
    start_time = time.time()

    # Parse the QSO sequence
    spec_indices = parse_qso_sequence(args.n_qso)

    # Run the convolution method in parallel
    results = parallel_convolution_method_absorber_finder_QSO_spectra(
        args.input_fits_file, spec_indices, absorber=args.absorber,
        ker_width_pix=constants.search_parameters[args.absorber]["ker_width_pixels"],
        coeff_sigma=constants.search_parameters[args.absorber]["coeff_sigma"],
        mult_resi=constants.search_parameters[args.absorber]["mult_resi"],
        d_pix=constants.search_parameters[args.absorber]["d_pix"],
        pm_pixel=constants.search_parameters[args.absorber]["pm_pixel"],
        sn_line1=constants.search_parameters[args.absorber]["sn1_thresh"],
        sn_line2=constants.search_parameters[args.absorber]["sn2_thresh"],
        use_covariance=False, n_jobs=args.n_tasks * args.ncpus
    )

    # Save the results to a FITS file
    save_results_to_fits(results, args.output, headers, args.absorber)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
