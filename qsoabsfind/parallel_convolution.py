"""
This script contains a function that runs the absorber finder in parallel for many spectra.
"""

import numpy as np
import argparse
import time
import os
from multiprocessing import Pool
from .absfinder import read_single_spectrum_and_find_absorber
from .io import save_results_to_fits
from .utils import read_nqso_from_header, get_package_versions, parse_qso_sequence

def run_convolution_method_absorber_finder_QSO_spectra(fits_file, spec_index, absorber, kwargs):
    """
    Wrapper function to unpack parameters and call the main convolution method.

    Args:
        fits_file (str): Path to the FITS file containing Normalized QSO spectra.
        spec_indices (list or numpy.array): Indices of quasars in the data matrix.
        absorber (str): Absorber name for searching doublets (MgII, CIV). Default is 'MgII'.
        kwargs (dict): search parameters as described in qsoabsfind.constants()

    Returns:
        tuples containing detected absorber details

    """
    return read_single_spectrum_and_find_absorber(fits_file, spec_index, absorber, **kwargs)

def parallel_convolution_method_absorber_finder_QSO_spectra(fits_file, spec_indices, absorber, n_jobs, **kwargs):
    """
    Run convolution_method_absorber_finder_in_QSO_spectra in parallel using
    multiprocessing.

    Args:
        fits_file (str): Path to the FITS file containing Normalized QSO spectra.
        spec_indices (list or numpy.array): Indices of quasars in the data matrix.
        absorber (str): Absorber name for searching doublets (MgII, CIV). Default is 'MgII'.
        n_jobs (int): Number of parallel jobs to run.
        kwargs (dict): search parameters as described in qsoabsfind.constants()

    Returns:
        dict: A dictionary containing combined results from all parallel runs.
    """

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
        print(f"INFO: Using user-provided constants from: {os.path.abspath(args.constant_file)}")
        print("INFO: Overwriting QSO_CONSTANTS_FILE variable with this path")
        os.environ['QSO_CONSTANTS_FILE'] = os.path.abspath(args.constant_file)

    print(f"INFO: QSO_CONSTANTS_FILE: {os.environ['QSO_CONSTANTS_FILE']}")
    # set the new constants
    from .config import load_constants
    constants = load_constants()

    # Prepare headers
    headers = {}
    for header in args.headers:
        key, value = header.split('=')
        headers[key] = {"value": value, "comment": ""}

    # Add search parameters and package versions to headers
    headers.update({
        'ABSORBER': {"value": args.absorber, "comment": 'Absorber name'},
        'KERWIDTH': {"value": str(constants.search_parameters[args.absorber]["ker_width_pixels"]), "comment": 'Kernel width in pixels (ker_width_pixels)'},
        'COEFFSIG': {"value": constants.search_parameters[args.absorber]["coeff_sigma"], "comment": 'sigma threshold (coeff_sigma)'},
        'MULTRE': {"value": constants.search_parameters[args.absorber]["mult_resi"], "comment": 'Multiplicative factor for residuals (mult_resi)'},
        'D_PIX': {"value": constants.search_parameters[args.absorber]["d_pix"], "comment": 'tolerance for line separation (in Ang) (d_pix)'},
        'PM_PIXEL': {"value": constants.search_parameters[args.absorber]["pm_pixel"], "comment": 'N_Pixel for noise estimation (pm_pixel)'},
        'SN_LINE1': {"value": constants.search_parameters[args.absorber]["sn_line1"], "comment": 'S/N threshold for first line (sn_line1)'},
        'SN_LINE2': {"value": constants.search_parameters[args.absorber]["sn_line2"], "comment": 'S/N threshold for second line (sn_line2)'},
        'EWCOVAR': {"value": constants.search_parameters[args.absorber]["use_covariance"], "comment": 'Use covariance for EW error (use_covariance)'},
        'LOGWAVE': {"value": constants.search_parameters[args.absorber]["logwave"], "comment": 'Use log wavelength scaling (logwave)'}, 'LAM_ESEP': {"value": constants.search_parameters[args.absorber]["lam_edge_sep"], "comment": 'wavelength edges to avoid noisy regions'}
    })

    package_versions = get_package_versions()
    for pkg, ver in package_versions.items():
        headers[pkg.upper()] = {"value": ver, "comment": f'{pkg} version'}
    headers['QSOABFI'] = headers.pop('QSOABSFIND')
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
        n_jobs=args.n_tasks * args.ncpus, **constants.search_parameters[args.absorber]
    )

    # only save absorber file if there at least one absorber is detected
    if len(results["index_spec"])>0:
        # Save the results to a FITS file
        save_results_to_fits(results, args.input_fits_file, args.output, headers, args.absorber)
    else:
        print(f'INFO: No {args.absorber} absorbers found, no file saved..')

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
