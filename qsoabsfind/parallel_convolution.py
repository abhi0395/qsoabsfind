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
from astropy.table import Table
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


def _init_worker_logging(queue, level=logging.WARNING):
    """Route all worker-process log records (including captured warnings) through the main-process queue."""
    root = logging.getLogger()
    root.handlers = []
    root.addHandler(logging.handlers.QueueHandler(queue))
    # Keep root at WARNING so third-party libraries (numba, scipy, …) stay quiet.
    # Raise only qsoabsfind's own logger to the requested level.
    root.setLevel(logging.WARNING)
    logging.getLogger('qsoabsfind').setLevel(level)
    logging.captureWarnings(True)

def run_convolution_method_absorber_finder_QSO_spectra(fits_file, spec_index, absorber, kwargs):
    """
    Wrapper function to unpack parameters and call the main convolution method.

    Args:
        fits_file (str): Path to the FITS file containing Normalized QSO spectra.
        spec_index (int): Index of the quasar spectrum to retrieve from the FITS file.
        absorber (str): Absorber name for searching doublets (MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, NaI). Default is 'MgII'.
        kwargs (dict): search parameters as described in data/desi/desi_constants.py

    Returns:
        dict: Detected absorber details from single-spectrum run.

    """
    constant_file = kwargs.pop('constant_file', None)

    return read_single_spectrum_and_find_absorber(fits_file, spec_index,
                                absorber,
                                constant_file=constant_file,
                                **kwargs)

def _run_single_job(params):
    """Unpack tuple params for imap-based iteration."""
    return run_convolution_method_absorber_finder_QSO_spectra(*params)

def parallel_convolution_search(
    fits_file, spec_indices, absorber, n_jobs, warnings_file=None, zabs_known_map=None, constant_file=None, **kwargs
):
    """
    Run convolution_method_absorber_finder_in_QSO_spectra in parallel using
    multiprocessing.

    Args:
        fits_file (str): Path to the FITS file containing Normalized QSO spectra.
        spec_indices (list or numpy.ndarray): Indices of quasars in the data matrix.
        absorber (str): Absorber name for searching doublets (MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, NaI).
        n_jobs (int): Number of parallel jobs to run.
        warnings_file (str, optional): Path to a file where worker-process warnings are written. Default is None.
        zabs_known_map (dict, optional): Mapping of spec_index (int) to a list of known absorber
            redshifts. When provided, the convolution search is skipped for those spectra and
            only Gaussian fitting and selection are run at the supplied redshifts. Spectra that
            do not appear in the map are searched in the normal way. Default is None.
        constant_file (str): constant file for the search parameters (*.py), default is None
        **kwargs: Search parameters as described in qsoabsfind.constants().

    Returns:
        dict: Combined results from all spectra, with only absorbers with z_abs > 0 retained.
            Keys: ``index_spec``, ``z_abs``, ``gauss_fit``, ``gauss_fit_std``,
            ``ew_1_mean``, ``ew_2_mean``, ``ew_total_mean``, ``ew_1_error``,
            ``ew_2_error``, ``ew_total_error``, ``z_abs_err``, ``sn_1``, ``sn_2``,
            ``vel_disp1``, ``vel_disp2``, ``delta_chi2_line1``, ``delta_chi2_line2``.
    """


    if zabs_known_map is not None:
        params_list = []
        for spec_index in spec_indices:
            spec_kwargs = dict(kwargs)
            zk = zabs_known_map.get(int(spec_index))
            if zk is not None:
                spec_kwargs['zabs_known'] = zk
            params_list.append((fits_file, spec_index, absorber, spec_kwargs))
    else:
        if constant_file is not None:
            kwargs['constant_file'] = constant_file
        params_list = [(fits_file, spec_index, absorber, kwargs) for spec_index in spec_indices]

    # Run jobs in parallel with live progress bar (ordered, streamed results).
    # Warnings are routed in a separate log file.

    verbose = kwargs.get('verbose', False)
    worker_level = logging.DEBUG if verbose else logging.WARNING

    pool_kwargs = {"processes": n_jobs}
    listener = None
    if warnings_file:
        _log_fmt = logging.Formatter(
            '[%(asctime)s] %(levelname)s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
        )
        _warn_queue = multiprocessing.Queue(-1)
        _warn_handler = logging.FileHandler(warnings_file, encoding='utf-8')
        _warn_handler.setFormatter(_log_fmt)
        listener_handlers = [_warn_handler]
        if verbose:
            _stream_handler = logging.StreamHandler()
            _stream_handler.setFormatter(_log_fmt)
            _stream_handler.setLevel(logging.DEBUG)
            listener_handlers.append(_stream_handler)
        listener = logging.handlers.QueueListener(
            _warn_queue, *listener_handlers, respect_handler_level=True
        )
        listener.start()
        pool_kwargs["initializer"] = _init_worker_logging
        pool_kwargs["initargs"] = (_warn_queue, worker_level)

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
        'delta_chi2_line1': [],
        'delta_chi2_line2': [],
        'pure_redchi2': [],
        'unsearchable_indices': [],
        'snr_qso_map': {},
    }
    if zabs_known_map is not None:
        combined_results['zabs_known'] = []

    for result in results:
        # in known-z mode keep every row (z_abs=-1, 0, or a fitted value);
        # in convolution mode keep only detected absorbers (z_abs > 0)
        if 'zabs_known' in combined_results:
            keep = np.ones(len(result['z_abs']), dtype=bool)
        else:
            keep = np.array(result['z_abs']) > 0
        if np.all(np.array(result['z_abs']) == -1):
            combined_results['unsearchable_indices'].append(int(result['index_spec'][0]))
        combined_results['snr_qso_map'][int(result['index_spec'][0])] = float(result.get('snr_qso', -1.0))

        combined_results['index_spec'].extend(np.array(result['index_spec'])[keep])
        combined_results['z_abs'].extend(np.array(result['z_abs'])[keep])
        combined_results['gauss_fit'].extend(np.array(result['gauss_fit'])[keep])
        combined_results['gauss_fit_std'].extend(np.array(result['gauss_fit_std'])[keep])
        combined_results['ew_1_mean'].extend(np.array(result['ew_1_mean'])[keep])
        combined_results['ew_2_mean'].extend(np.array(result['ew_2_mean'])[keep])
        combined_results['ew_total_mean'].extend(np.array(result['ew_total_mean'])[keep])
        combined_results['ew_1_error'].extend(np.array(result['ew_1_error'])[keep])
        combined_results['ew_2_error'].extend(np.array(result['ew_2_error'])[keep])
        combined_results['ew_total_error'].extend(np.array(result['ew_total_error'])[keep])
        combined_results['z_abs_err'].extend(np.array(result['z_abs_err'])[keep])
        combined_results['sn_1'].extend(np.array(result['sn_1'])[keep])
        combined_results['sn_2'].extend(np.array(result['sn_2'])[keep])
        combined_results['vel_disp1'].extend(np.array(result['vel_disp1'])[keep])
        combined_results['vel_disp2'].extend(np.array(result['vel_disp2'])[keep])
        combined_results['delta_chi2_line1'].extend(np.array(result['delta_chi2_line1'])[keep])
        combined_results['delta_chi2_line2'].extend(np.array(result['delta_chi2_line2'])[keep])
        combined_results['pure_redchi2'].extend(np.array(result['pure_redchi2'])[keep])

        if 'zabs_known' in combined_results:
            zk = result.get('zabs_known')
            if zk is not None:
                combined_results['zabs_known'].extend(np.array(zk)[keep])
            else:
                combined_results['zabs_known'].extend([np.nan] * int(keep.sum()))

    return combined_results

def main():
    parser = argparse.ArgumentParser(
        description='Parallelized convolution-based method to detect metal doublets in SDSS/DESI-like low-resolution quasar spectra using adaptive S/N.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--config', type=str, default=None, help='Path to a YAML config file. All keys must match CLI argument names (underscores). CLI flags always override YAML values.')
    parser.add_argument('--input-fits-file', type=str, required=False, help='Path to the input FITS file, containing residual spectra.')
    parser.add_argument('--n-qso', type=str, required=False, help="Number of QSO spectra to process, or a bash-like sequence (e.g., '100', '1-1000', '1-1000:10'). If not provided, code will run all the spectra")
    parser.add_argument('--absorber', type=str, required=False, help='Name of the absorber doublet to search for. Built-in search windows are available for MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, and NaI; custom doublets are also supported.')
    parser.add_argument('--constant-file', type=str, help='Path to the constants .py file, please follow the exact same structure as described in the documentation.')
    parser.add_argument('--output', type=str, required=False, help='Path to the output FITS file to save absorber catalog.')
    parser.add_argument('--headers', type=str, nargs='+', help='Headers for the output FITS file in the format NAME=VALUE.')
    parser.add_argument('--ncpus', type=int, required=False, default=4, help='Number of CPUs for parallel processing.')
    parser.add_argument('--coldens-dv', type=float, required=False, default=None,
        help='If provided, also compute total column densities using the apparent optical depth method '
             '(AODM; Savage & Sembach 1991). The value sets the +/- velocity range (km/s) for '
             'optical-depth integration around each line centre (e.g. 300). '
             'Adds a COLUMN_DENSITY HDU to the output file.')
    parser.add_argument('--verbose', action='store_true', help='Enable detailed per-spectrum/debug logging.')
    parser.add_argument('--zabs-known-file', type=str, default=None,
        help='Path to a FITS file with columns INDEX_SPEC and Z_ABS. When provided, the '
             'convolution search is skipped for the listed spectra and only Gaussian fitting '
             'is run at the supplied redshifts. Multiple rows with the same INDEX_SPEC are '
             'treated as multiple known redshifts for that spectrum.')
    parser.add_argument('--trapz-ew-sigma', type=float, default=None,
        help='If provided, equivalent widths in the output catalog are measured using '
             'trapezoidal integration over a window of +/- TRAPZ_EW_SIGMA * sigma around '
             'each line centre (sigma from the Gaussian fit). These EWs are used '
             'consistently for both the stored catalog columns and the absorber selection '
             'criteria (ew_snr, doublet ratio). Gaussian fit parameters are always retained. '
             'Default: use Gaussian analytic EW.')

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
        raise ValueError(
            f"Absorber '{args.absorber}' not found in doublet_keys. "
            f"Built-in absorbers: {list(doublet_keys.keys())}. "
            "To use a custom doublet, add it to your constants file "
            "(see docs/paramfile.rst for the required format)."
        )

    logger.info('User provided arguments and constants loaded')
    for key, value in vars(args).items():
        logger.info("%s: %s", key, value)
    for key, value in user_constants.search_parameters.items():
        logger.info("%s: %s", key, value)

    user_constants.search_parameters["verbose"] = args.verbose

    # Patch qsoabsfind.constants in-place with any overrides from the user constants file.
    # All modules that access constants via `from . import constants as _constants` (i.e.
    # absorberutils and absfinder) will automatically see the updated values -- no function
    # signature changes needed.
    from . import constants as _pkg_constants
    logger.info('Physical constant resolution (user file overrides shown with *):')
    for _name in _pkg_constants.OVERRIDABLE_CONSTANTS:
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

    if args.coldens_dv is not None:
        logger.info('Will also calculate column densities using apparent optical depth method (AODM)')
        headers.update({
                'N_METHOD': {
                    'value': 'AODM',
                    'comment': 'Column Density Method: apparent optical depth'
                },
                'DELTA_V': {
                    'value': args.coldens_dv,
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

    n_qso_explicit = args.n_qso  # None if user did not pass --n-qso
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

    if args.trapz_ew_sigma is not None:
        user_constants.search_parameters['trapz_ew_sigma'] = args.trapz_ew_sigma
        logger.info('Trapezoidal EW method enabled with n_sigma = %s', args.trapz_ew_sigma)
    elif 'trapz_ew_sigma' not in user_constants.search_parameters:
        user_constants.search_parameters['trapz_ew_sigma'] = None

    # Load known-redshift map if the user provided a FITS file
    zabs_known_map = None
    if args.zabs_known_file:

        zk_table = Table.read(args.zabs_known_file)
        if 'INDEX_SPEC' not in zk_table.colnames or 'Z_ABS' not in zk_table.colnames:
            raise ValueError(
                f"--zabs-known-file must contain columns INDEX_SPEC and Z_ABS, "
                f"found: {zk_table.colnames}")
        zabs_known_map = {}
        for row in zk_table:
            idx = int(row['INDEX_SPEC'])
            zabs_known_map.setdefault(idx, []).append(float(row['Z_ABS']))
        logger.info('Loaded %d known-redshift entries for %d spectra from %s',
                    len(zk_table), len(zabs_known_map), args.zabs_known_file)
        spec_indices = sorted(zabs_known_map.keys())
        if n_qso_explicit is not None:
            n_qso_str = str(n_qso_explicit)
            if '-' in n_qso_str:
                # Range-style (e.g. '1-100'): keep only those indices inside the range
                requested_set = set(parse_qso_sequence(n_qso_explicit))
                spec_indices = [i for i in spec_indices if i in requested_set]
            else:
                # Count-style (e.g. '10'): take the first N entries from the known file
                n_cap = int(n_qso_str)
                spec_indices = spec_indices[:n_cap]
            logger.info('After --n-qso %s filter: %d spectra to process', n_qso_explicit, len(spec_indices))
        else:
            logger.info('Running only on %d spectra listed in the known-redshift file', len(spec_indices))

    # Run the convolution method in parallel

    results = parallel_convolution_search(
        args.input_fits_file, spec_indices, absorber=args.absorber,
        n_jobs=n_jobs, warnings_file=warnings_file,
        zabs_known_map=zabs_known_map, constant_file=const_path, **user_constants.search_parameters
    )

    # only save absorber file if there at least one absorber is detected
    if len(results["index_spec"])>0:
        # Save the results to a FITS file
        if zabs_known_map is not None:
            n_valid = int(np.sum(np.array(results["z_abs"]) > 0))
            logger.info('Number of %s systems validated (z_abs > 0): %s of %s entries',
                        args.absorber, n_valid, len(results["index_spec"]))
        else:
            logger.info('Number of %s systems found: %s', args.absorber, len(results["index_spec"]))
        save_results_to_fits(results, args.input_fits_file, args.output, headers, args.absorber,
                             spec_indices=spec_indices)

        if args.coldens_dv is not None:
            logwave = user_constants.search_parameters["logwave"]
            col_tt = return_total_column_density_table(args.input_fits_file, args.absorber, args.output, user_constants.search_parameters["continuum_error_frac"], args.coldens_dv, logwave, n_jobs)
            append_table_to_fits(args.output, col_tt, 'COLUMN_DENSITY')
    else:
        logger.info('No %s absorbers found, no file saved', args.absorber)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    logger.info("Elapsed time: %.2f seconds", elapsed_time)
    logger.info("Script ended at: %s", datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

if __name__ == "__main__":
    main()
