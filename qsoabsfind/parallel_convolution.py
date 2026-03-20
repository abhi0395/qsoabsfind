"""
This script contains a function that runs the absorber
finder in parallel for many spectra.
"""
import argparse
import time
import os
import logging
import logging.handlers
import multiprocessing
from multiprocessing import Pool
from datetime import datetime
import numpy as np
from tqdm import tqdm
from .absfinder import read_single_spectrum_and_find_absorber
from .columndensity import return_total_column_density_table
from .io import append_table_to_fits
from .io import save_results_to_fits
from .absorberutils import return_search_window_wavelength_range
from .utils import read_nqso_from_header, get_package_versions, parse_qso_sequence, update_header
from .constants import doublet_keys
from .logger import setup_logging
from .config import load_yaml_config

logger = logging.getLogger(__name__)


def _init_worker_logging(queue):
    """Route all worker-process log records (including captured warnings) through the main-process queue."""
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(logging.handlers.QueueHandler(queue))
    root.setLevel(logging.WARNING)
    logging.captureWarnings(True)

def run_convolution_method_absorber_finder_QSO_spectra(fits_file, spec_index, absorber, kwargs):
    """
    Wrapper function to unpack parameters and call the main convolution method.

    Args:
        fits_file (str): Path to the FITS file containing Normalized QSO spectra.
        spec_index (int): Index of the quasar spectrum to retrieve from the FITS file.
        absorber (str): Absorber name for searching doublets (MgII, CIV, OVI, NV, SiIV, AlIII, FeII). Default is 'MgII'.
        kwargs (dict): search parameters as described in data/desi/desi_constants.py

    Returns:
        dict: Detected absorber details from single-spectrum run.

    """
    return read_single_spectrum_and_find_absorber(fits_file, spec_index, absorber, **kwargs)


def _run_single_job(params):
    """Unpack tuple params for imap-based iteration."""
    return run_convolution_method_absorber_finder_QSO_spectra(*params)

def parallel_convolution_search(
    fits_file, spec_indices, absorber, n_jobs, warnings_file=None, **kwargs
):
    """
    Run convolution_method_absorber_finder_in_QSO_spectra in parallel using
    multiprocessing.

    Args:
        fits_file (str): Path to the FITS file containing Normalized QSO spectra.
        spec_indices (list or numpy.ndarray): Indices of quasars in the data matrix.
        absorber (str): Absorber name for searching doublets (MgII, CIV, OVI, NV, SiIV, AlIII, FeII).
        n_jobs (int): Number of parallel jobs to run.
        warnings_file (str, optional): Path to a file where worker-process warnings are written. Default is None.
        **kwargs: Search parameters as described in qsoabsfind.constants().

    Returns:
        dict: Combined results from all spectra, with only absorbers with z_abs > 0 retained.
            Keys: ``index_spec``, ``z_abs``, ``gauss_fit``, ``gauss_fit_std``,
            ``ew_1_mean``, ``ew_2_mean``, ``ew_total_mean``, ``ew_1_error``,
            ``ew_2_error``, ``ew_total_error``, ``z_abs_err``, ``sn_1``, ``sn_2``,
            ``vel_disp1``, ``vel_disp2``, ``delta_chi2``.
    """

    params_list = [(fits_file, spec_index, absorber, kwargs) for spec_index in spec_indices]

    # Run jobs in parallel with live progress bar (ordered, streamed results).
    # Worker warnings are routed through a QueueHandler so all writes to
    # warnings_file are serialised by the main-process QueueListener.
    pool_kwargs = {"processes": n_jobs}
    listener = None
    if warnings_file:
        _warn_queue = multiprocessing.Queue(-1)
        _warn_handler = logging.FileHandler(warnings_file, encoding='utf-8')
        _warn_handler.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        ))
        listener = logging.handlers.QueueListener(
            _warn_queue, _warn_handler, respect_handler_level=True
        )
        listener.start()
        pool_kwargs["initializer"] = _init_worker_logging
        pool_kwargs["initargs"] = (_warn_queue,)

    try:
        with Pool(**pool_kwargs) as pool:
            results_iter = pool.imap(_run_single_job, params_list)
            results = list(
                tqdm(
                    results_iter,
                    total=len(params_list),
                    desc=f'{absorber} search',
                    unit='spec',
                )
            )
    finally:
        if listener:
            listener.stop()

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
        valid_indices = np.array(result['z_abs']) > 0

        combined_results['index_spec'].extend(np.array(result['index_spec'])[valid_indices])
        combined_results['z_abs'].extend(np.array(result['z_abs'])[valid_indices])
        combined_results['gauss_fit'].extend(np.array(result['gauss_fit'])[valid_indices])
        combined_results['gauss_fit_std'].extend(np.array(result['gauss_fit_std'])[valid_indices])
        combined_results['ew_1_mean'].extend(np.array(result['ew_1_mean'])[valid_indices])
        combined_results['ew_2_mean'].extend(np.array(result['ew_2_mean'])[valid_indices])
        combined_results['ew_total_mean'].extend(np.array(result['ew_total_mean'])[valid_indices])
        combined_results['ew_1_error'].extend(np.array(result['ew_1_error'])[valid_indices])
        combined_results['ew_2_error'].extend(np.array(result['ew_2_error'])[valid_indices])
        combined_results['ew_total_error'].extend(np.array(result['ew_total_error'])[valid_indices])
        combined_results['z_abs_err'].extend(np.array(result['z_abs_err'])[valid_indices])
        combined_results['sn_1'].extend(np.array(result['sn_1'])[valid_indices])
        combined_results['sn_2'].extend(np.array(result['sn_2'])[valid_indices])
        combined_results['vel_disp1'].extend(np.array(result['vel_disp1'])[valid_indices])
        combined_results['vel_disp2'].extend(np.array(result['vel_disp2'])[valid_indices])
        combined_results['delta_chi2'].extend(np.array(result['delta_chi2'])[valid_indices])

    return combined_results

def main():
    parser = argparse.ArgumentParser(
        description='Parallelized convolution-based method to detect metal doublets in SDSS/DESI-like low-resolution quasar spectra using adaptive S/N.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, default=None, help='Path to a YAML config file. All keys must match CLI argument names (underscores). CLI flags always override YAML values.')
    parser.add_argument('--input-fits-file', type=str, required=False, help='Path to the input FITS file, containing residual spectra.')
    parser.add_argument('--n-qso', type=str, required=False, help="Number of QSO spectra to process, or a bash-like sequence (e.g., '100', '1-1000', '1-1000:10'). If not provided, code will run all the spectra")
    parser.add_argument('--absorber', type=str, required=False, help='Absorber name for searching doublets (options: MgII, CIV, OVI, NV, SiIV, AlIII, FeII).')
    parser.add_argument('--constant-file', type=str, help='Path to the constants .py file, please follow the exact same structure as described in the documentation.')
    parser.add_argument('--output', type=str, required=False, help='Path to the output FITS file to save absorber catalog.')
    parser.add_argument('--headers', type=str, nargs='+', help='Headers for the output FITS file in the format NAME=VALUE.')
    parser.add_argument('--ncpus', type=int, required=False, default=4, help='Number of CPUs for parallel processing.')
    parser.add_argument('--coldens', default=False, required=False, action="store_true", help='If provided, code will also calculate total column densities using apparent optical depth method')
    parser.add_argument('--dv', type=float, required=False, default=300, help='if --coldens is provided, +/- |dv| range (in km/s) will be used to calculate optical depth around each line, default: 300 km/s')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed per-spectrum/debug logging.')

    # --- Two-pass parse: load YAML defaults first, CLI args override them ---
    # First pass: extract --config without failing on unknown/required args
    pre_args, _ = parser.parse_known_args()
    if pre_args.config:
        yaml_defaults = load_yaml_config(pre_args.config)
        parser.set_defaults(**yaml_defaults)

    args = parser.parse_args()

    # Validate required args (may come from YAML or CLI)
    missing = [name for name, val in [
        ('--input-fits-file', args.input_fits_file),
        ('--absorber', args.absorber),
        ('--output', args.output),
        ('--constant-file', args.constant_file),
    ] if not val]
    if missing:
        parser.error(f"The following required arguments are missing (provide via CLI or --config): {', '.join(missing)}")

    # all runtime, optimization, and warning logs will be written to a log file in the output directory
    output_dir = os.path.dirname(os.path.abspath(args.output))
    warnings_file = os.path.join(output_dir, 'warnings.log')
    setup_logging(verbose=args.verbose, warnings_file=warnings_file)

    logger.info("\n\nScript started at: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S\n'))
    logger.info("\n\nWarnings logged to: %s\n", warnings_file)

    # Read search parameters from user-provided file
    if args.constant_file and os.path.isfile(args.constant_file):
        const_path = os.path.abspath(args.constant_file)
        logger.info("Using user-provided constants from: %s", const_path)
    else:
        raise FileNotFoundError(f"ERROR: Provided constants file does not exist: {const_path}")

    # Load constants
    from .config import load_constants
    user_constants = load_constants(const_path)

    if args.absorber not in doublet_keys:
        raise ValueError(f"ERROR: Unsupported absorber, it must be from {doublet_keys.keys()}")

    logger.info('User provided arguments and constants loaded')
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)
    for key, value in user_constants.search_parameters.items():
        logger.info("%s: %s", key, value)

    user_constants.search_parameters["verbose"] = args.verbose

    # Patch qsoabsfind.constants in-place with any overrides from the user constants file.
    # All modules that access constants via `from . import constants as _constants` (i.e.
    # absorberutils and absfinder) will automatically see the updated values — no function
    # signature changes needed.
    from . import constants as _pkg_constants
    _overridable = ('SMALL_WAVE', 'LARGE_WAVE', 'LAM_CIV_MIN', 'MIN_NPIXEL')
    logger.info('Physical constant resolution (user file overrides shown with *):')
    for _name in _overridable:
        _user_val = getattr(user_constants, _name, None)
        _pkg_val  = getattr(_pkg_constants, _name)
        if _user_val is not None and _user_val != _pkg_val:
            logger.info('%-15s = %s  (overrides package default: %s)', _name, _user_val, _pkg_val)
            setattr(_pkg_constants, _name, _user_val)
        else:
            logger.info('%-15s = %s  (package default)', _name, _pkg_val)

    lam_blue, lam_red = return_search_window_wavelength_range(
        args.absorber,
        user_constants.search_parameters["start_rest_wave"],
        user_constants.search_parameters["end_rest_wave"],
        verbose=args.verbose
    )

    user_constants.search_parameters["lam_blue"] = lam_blue
    user_constants.search_parameters["lam_red"] = lam_red

    headers = update_header(args, user_constants)

    if args.coldens:
        logger.info('Will also calculate column densities using apparent optical depth method (AODM)')
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
        logger.info('Total quasars found in the input file = %s, will run on all of them', args.n_qso)
    # Parse the QSO sequence
    spec_indices = parse_qso_sequence(args.n_qso)

    # define number of CPUs cores
    n_jobs = min(args.ncpus, max(1, multiprocessing.cpu_count() - 1)) ## getting some CPUs for safe I/O processing
    logger.info('number of CPUs used = %s', n_jobs)

    nboot = user_constants.search_parameters.get("nboot")
    if "nboot" not in user_constants.search_parameters:
        user_constants.search_parameters["nboot"] = None

    if nboot is not None and nboot>0:
        logger.info('Gaussian fitting parameter estimation will be done with %s bootstrapping iterations', nboot)

    # Run the convolution method in parallel
    results = parallel_convolution_search(
        args.input_fits_file, spec_indices, absorber=args.absorber,
        n_jobs=n_jobs, warnings_file=warnings_file, **user_constants.search_parameters
    )

    # only save absorber file if there at least one absorber is detected
    if len(results["index_spec"])>0:
        # Save the results to a FITS file
        logger.info('Number of %s systems found: %s', args.absorber, len(results["index_spec"]))
        save_results_to_fits(results, args.input_fits_file, args.output, headers, args.absorber)
    else:
        logger.info('No %s absorbers found, no file saved', args.absorber)

    if args.coldens:
        logwave = user_constants.search_parameters["logwave"]
        col_tt = return_total_column_density_table(args.input_fits_file, args.absorber, args.output, user_constants.search_parameters["continuum_error_frac"], args.dv, logwave, n_jobs)
        append_table_to_fits(args.output, col_tt, 'COLUMN_DENSITY')

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    logger.info("Elapsed time: %.2f seconds", elapsed_time)
    logger.info("Script ended at: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == "__main__":
    main()
