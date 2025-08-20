"""
This script contains a function that runs the absorber finder in parallel for many spectra.
"""
import argparse
import time
import os
import multiprocessing
from multiprocessing import Pool
from datetime import datetime
import numpy as np
from .absfinder import read_single_spectrum_and_find_absorber
from .columndensity import return_total_column_density_table
from .io import append_table_to_fits
from .io import save_results_to_fits
from .absorberutils import return_search_window_wavelength_range
from .utils import read_nqso_from_header, get_package_versions, parse_qso_sequence, update_header
from .constants import doublet_keys

def run_convolution_method_absorber_finder_QSO_spectra(fits_file, spec_index, absorber, kwargs):
    """
    Wrapper function to unpack parameters and call the main convolution method.

    Args:
        fits_file (str): Path to the FITS file containing Normalized QSO spectra.
        spec_indices (list or numpy.array): Indices of quasars in the data matrix.
        absorber (str): Absorber name for searching doublets (MgII, CIV, OVI, NV, SiIV, AlIII, FeII). Default is 'MgII'.
        kwargs (dict): search parameters as described in data/desi/desi_constants.py

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
        absorber (str): Absorber name for searching doublets (MgII, CIV, OVI, NV, SiIV, AlIII, FeII). Default is 'MgII'.
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
        'vel_disp1': [],
        'vel_disp2': [],
        'delta_chi2': [],
    }

    for result in results:
        (index_spec, z_abs, gauss_fit, gauss_fit_std, ew_1_mean, ew_2_mean, ew_total_mean,
         ew_1_error, ew_2_error, ew_total_error, z_abs_err, sn_1, sn_2, vel_disp1, vel_disp2, delta_chi2_array) = result

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
        combined_results['vel_disp1'].extend(np.array(vel_disp1)[valid_indices])
        combined_results['vel_disp2'].extend(np.array(vel_disp2)[valid_indices])
        combined_results['delta_chi2'].extend(np.array(delta_chi2_array)[valid_indices])

    return combined_results

def main():
    parser = argparse.ArgumentParser(description='Parallelized convolution-based method to detect metal doublets in SDSS/DESI-like low-resolution quasar spectra using adaptive S/N.')
    parser.add_argument('--input-fits-file', type=str, required=True, help='Path to the input FITS file, containing residual spectra.')
    parser.add_argument('--n-qso', type=str, required=False, help="Number of QSO spectra to process, or a bash-like sequence (e.g., '100', '1-1000', '1-1000:10'). If not provided, code will run all the spectra")
    parser.add_argument('--absorber', type=str, required=True, help='Absorber name for searching doublets (options: MgII, CIV, OVI, NV, SiIV, AlIII, FeII).')
    parser.add_argument('--constant-file', type=str, help='Path to the constants .py file, please follow the exact same structure as qsoabsfind.constants, i.e the default parameter that the code uses')
    parser.add_argument('--output', type=str, required=True, help='Path to the output FITS file to save absorber catalog.')
    parser.add_argument('--headers', type=str, nargs='+', help='Headers for the output FITS file in the format NAME=VALUE.')
    parser.add_argument('--ncpus', type=int, required=False, default=4, help='Number of CPUs for parallel processing.')
    parser.add_argument('--coldens', default=False, required=False, action="store_true", help='If provided, code will also calculate total column densities using apparent optical depth method')
    parser.add_argument('--dv', type=float, required=False, default=300, help='if --coldens is provided, +/- |dv| range (in km/s) will be used to calculate optical depth around each line, default: 300 km/s')

    print(f"\nINFO: Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("==========\n")
    args = parser.parse_args()

    # Read search parameters from user-provided file
    if args.constant_file and os.path.abspath(args.constant_file):
        const_path = os.path.abspath(args.constant_file)
        print(f"INFO: Using user-provided constants from: {const_path}")
    else:
        raise FileNotFoundError(f"ERROR: Provided constants file does not exist: {const_path}")

    # Load constants
    from .config import load_constants
    user_constants = load_constants(const_path)

    if args.absorber not in doublet_keys:
        raise ValueError(f"ERROR: Unsupported absorber, it must be from {doublet_keys.keys()}")

    lam_blue, lam_red = return_search_window_wavelength_range(args.absorber, user_constants.search_parameters["start_rest_wave"], user_constants.search_parameters["end_rest_wave"])

    user_constants.search_parameters["lam_blue"] = lam_blue
    user_constants.search_parameters["lam_red"] = lam_red

    headers = update_header(args, user_constants)

    if args.coldens:
        print('INFO: Will also calculate column densities using apparent optical depth method (AODM)')
        headers.update({
                'N_METHOD': {
                    'value': 'AODM',
                    'comment': 'Column Density Method: apparent optical depth'
                },
                'DELTA_V': {
                    'value': args.dv,
                    'comment': '+/- velocity (km/s) to calculate optical depth'
                }
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

    # define number of CPUs cores
    n_jobs = min(args.ncpus, max(1, multiprocessing.cpu_count() - 1)) ## getting some CPUs for safe I/O processing
    print(f'INFO: number of CPUs used = {n_jobs}')

    if "nboot" not in user_constants.search_parameters:
        user_constants.search_parameters["nboot"] = None
    else:
        nboot = user_constants.search_parameters["nboot"]

    if nboot is not None and nboot>0:
        print(f'INFO: Gaussian fitting Parameter estimation will be done with {nboot} bootstrapping estimation')

    # Run the convolution method in parallel
    results = parallel_convolution_method_absorber_finder_QSO_spectra(
        args.input_fits_file, spec_indices, absorber=args.absorber,
        n_jobs=n_jobs, **user_constants.search_parameters
    )

    # only save absorber file if there at least one absorber is detected
    if len(results["index_spec"])>0:
        # Save the results to a FITS file
        print(f'INFO: Number of {args.absorber} systems found: {len(results["index_spec"])}')
        save_results_to_fits(results, args.input_fits_file, args.output, headers, args.absorber)
    else:
        print(f'INFO: No {args.absorber} absorbers found, no file saved..')

    if args.coldens:
        logwave = user_constants.search_parameters["logwave"]
        col_tt = return_total_column_density_table(args.input_fits_file, args.absorber, args.output, user_constants.search_parameters["continuum_error_frac"], args.dv, logwave, n_jobs)
        append_table_to_fits(args.output, col_tt, 'COLUMN_DENSITY')

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print(f"INFO: Script ended at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("===========\n")

if __name__ == "__main__":
    main()
