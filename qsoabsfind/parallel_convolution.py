"""
This script contains a function that runs the absorber finder in parallel for many spectra.
"""

import numpy as np
import argparse
import time
from multiprocessing import Pool
from .absfinder import read_single_spectrum_and_find_absorber
from .io import save_results_to_fits
import re
import os
import pkg_resources
from .config import load_constants
from .utils import read_nqso_from_header

def get_package_versions():
    """
    Get the versions of qsoabsfind and other relevant packages.

    Returns:
        dict: A dictionary containing the versions of the packages.
    """
    packages = ['qsoabsfind', 'numpy', 'astropy', 'scipy', 'numba', 'matplotlib']
    versions = {pkg: pkg_resources.get_distribution(pkg).version for pkg in packages}
    return versions


def run_convolution_method_absorber_finder_QSO_spectra(fits_file, spec_index, absorber, kwargs):
    """
    Wrapper function to unpack parameters and call the main convolution method.

    Args:
        fits_file (str): Path to the FITS file containing Normalized QSO spectra.
        spec_indices (list or numpy.array): Indices of quasars in the data matrix.
        absorber (str): Absorber name for searching doublets (MgII, CIV). Default is 'MgII'.
        kwargs (dictionary): search parameters as described in qsoabsfind.constants()

    Returns:
        tuples containing detected absorber details

    """
    return read_single_spectrum_and_find_absorber(fits_file, spec_index, absorber, **kwargs)

def parse_qso_sequence(qso_sequence):
    """
    Parse a bash-like sequence or a single integer to generate QSO indices.

    Args:
        qso_sequence (str or int): Bash-like sequence (e.g., '1-1000', '1-1000:10') or an integer.

    Returns:
        numpy.array: Array of QSO indices.
    """
    if isinstance(qso_sequence, int):
        return np.arange(qso_sequence)

    # Handle string input
    if isinstance(qso_sequence, str):
        if qso_sequence.isdigit():
            return np.arange(int(qso_sequence))

        match = re.match(r"(\d+)-(\d+)(?::(\d+))?", qso_sequence)
        if match:
            start, end, step = match.groups()
            start, end = int(start), int(end)
            step = int(step) if step else 1
            return np.arange(start, end + 1, step)

    # If none of the conditions matched, raise an error
    raise ValueError(f"Invalid QSO sequence format: '{qso_sequence}'. Use 'start-end[:step]' or an integer.")

def parallel_convolution_method_absorber_finder_QSO_spectra(fits_file, spec_indices, absorber='MgII', ker_width_pixels=[3, 4, 5, 6, 7, 8], coeff_sigma=2.5, mult_resi=1, d_pix=0.6, pm_pixel=200, sn_line1=3, sn_line2=2, use_covariance=False, logwave=True, verbose=False, n_jobs=1):
    """
    Run convolution_method_absorber_finder_in_QSO_spectra in parallel using
    multiprocessing.

    Args:
        fits_file (str): Path to the FITS file containing Normalized QSO spectra.
        spec_indices (list or numpy.array): Indices of quasars in the data matrix.
        absorber (str): Absorber name for searching doublets (MgII, CIV). Default is 'MgII'.
        ker_width_pixels (list): List of kernel widths in pixels. Default is [3, 4, 5, 6, 7, 8].
        coeff_sigma (float): Coefficient for sigma to apply threshold in the convolved array. Default is 2.5.
        mult_resi (float): Factor to shift the residual up or down. Default is 1.
        d_pix (float): Pixel distance for line separation during Gaussian fitting. Default is 0.6.
        pm_pixel (int): Pixel parameter for local noise estimation (default 200).
        sn_line1 (float): Signal-to-noise ratio for thresholding for line1.
        sn_line2 (float): Signal-to-noise ratio for thresholding for line2.
        use_covariance (bool): If want to use full covariance of scipy curve_fit for EW error calculation (default is False).
        logwave (bool): If wavelength on logscale (default True for SDSS).
        verbose (bool): if True will print lots of output for debugging
        n_jobs (int): Number of parallel jobs to run.

    Returns:
        dict: A dictionary containing combined results from all parallel runs.
    """

    kwargs = {'ker_width_pixels': ker_width_pixels, 'coeff_sigma': coeff_sigma, 'mult_resi': mult_resi, 'd_pix': d_pix, 'pm_pixel': pm_pixel, 'sn_line1': sn_line1, 'sn_line2': sn_line2, 'use_covariance': use_covariance, 'logwave': logwave, 'verbose': verbose}

    params_list = [(fits_file, spec_index, absorber, kwargs) for spec_index in spec_indices]

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
    parser = argparse.ArgumentParser(description='Run convolution-based adaptive S/N method to search for metal doublets in SDSS/DESI-like QSO spectra in parallel.')
    parser.add_argument('--input-fits-file', type=str, required=True, help='Path to the input FITS file.')
    parser.add_argument('--n-qso', type=str, required=False, help="Number of QSO spectra to process, or a bash-like sequence (e.g., '1-1000', '1-1000:10'). If not provided, code will run all the spectra")
    parser.add_argument('--absorber', type=str, required=True, help='Absorber name for searching doublets (MgII, CIV).')
    parser.add_argument('--constant-file', type=str, help='Path to the constants .py file, please follow the exact same structure as qsoabsfind.constants, i.e the default parameter that the code uses')
    parser.add_argument('--output', type=str, required=True, help='Path to the output FITS file.')
    parser.add_argument('--headers', type=str, nargs='+', help='Headers for the output FITS file in the format NAME=VALUE.')
    parser.add_argument('--n-tasks', type=int, required=True, help='Number of tasks.')
    parser.add_argument('--ncpus', type=int, required=True, help='Number of CPUs per task.')

    args = parser.parse_args()

    # Set the environment variable for the constants file
    if args.constant_file:
        if not os.path.isabs(args.constant_file):
            args.constant_file = os.path.abspath(os.path.join(os.getcwd(), args.constant_file))
            os.environ['QSO_CONSTANTS_FILE'] = args.constant_file
            print(f"INFO: Using user-provided constants from: {args.constant_file}")

    print(f"INFO: QSO_CONSTANTS_FILE: {os.environ['QSO_CONSTANTS_FILE']}")
    # set the new constants
    constants = load_constants()

    # Prepare headers
    headers = {}
    for header in args.headers:
        key, value = header.split('=')
        headers[key] = {"value": value, "comment": ""}

    # Add search parameters and package versions to headers
    headers.update({
        'ABSORBER': {"value": args.absorber, "comment": 'Absorber name'},
        'KERWIDTH': {"value": str(constants.search_parameters[args.absorber]["ker_width_pixels"]), "comment": 'Kernel width in pixels'},
        'COEFFSIG': {"value": constants.search_parameters[args.absorber]["coeff_sigma"], "comment": 'Coefficient for sigma threshold'},
        'MULTRE': {"value": constants.search_parameters[args.absorber]["mult_resi"], "comment": 'Multiplicative factor for residuals'},
        'D_PIX': {"value": constants.search_parameters[args.absorber]["d_pix"], "comment": 'toloerance for line separation (in Ang)'},
        'PM_PIXEL': {"value": constants.search_parameters[args.absorber]["pm_pixel"], "comment": 'Pixel parameter for local noise estimation'},
        'SN_LINE1': {"value": constants.search_parameters[args.absorber]["sn_line1"], "comment": 'S/N threshold for first line'},
        'SN_LINE2': {"value": constants.search_parameters[args.absorber]["sn_line2"], "comment": 'S/N threshold for second line'},
        'EWCOVAR': {"value": constants.search_parameters[args.absorber]["use_covariance"], "comment": 'Use covariance for EW error calculation'},
        'LOGWAVE': {"value": constants.search_parameters[args.absorber]["logwave"], "comment": 'Use log wavelength scaling'}
    })

    package_versions = get_package_versions()
    for pkg, ver in package_versions.items():
        headers[pkg.upper()] = {"value": ver, "comment": f'{pkg} version'}
    headers['QSABFI'] = headers.pop('QSOABSFIND')
    headers['MATPLOT'] = headers.pop('MATPLOTLIB')

    # Start timing
    start_time = time.time()

    if not args.n_qso:
        nqso = read_nqso_from_header(args.input_fits_file)
        args.n_qso = nqso
        print(f'INFO:: Total quasars found in the input file = {args.n_qso}, will run on all of them..')
    # Parse the QSO sequence
    spec_indices = parse_qso_sequence(args.n_qso)

    # Run the convolution method in parallel
    results = parallel_convolution_method_absorber_finder_QSO_spectra(
        args.input_fits_file, spec_indices, absorber=args.absorber,
        ker_width_pixels=constants.search_parameters[args.absorber]["ker_width_pixels"],
        coeff_sigma=constants.search_parameters[args.absorber]["coeff_sigma"],
        mult_resi=constants.search_parameters[args.absorber]["mult_resi"],
        d_pix=constants.search_parameters[args.absorber]["d_pix"],
        pm_pixel=constants.search_parameters[args.absorber]["pm_pixel"],
        sn_line1=constants.search_parameters[args.absorber]["sn_line1"],
        sn_line2=constants.search_parameters[args.absorber]["sn_line2"],
        use_covariance=constants.search_parameters[args.absorber]["use_covariance"],
        logwave=constants.search_parameters[args.absorber]["logwave"], verbose=constants.search_parameters[args.absorber]["verbose"], n_jobs=args.n_tasks * args.ncpus
    )

    # Save the results to a FITS file
    save_results_to_fits(results, args.input_fits_file, args.output, headers, args.absorber)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
