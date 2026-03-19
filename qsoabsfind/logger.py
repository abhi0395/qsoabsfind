"""
Centralised logging configuration for qsoabsfind.

Usage
-----
In each module, get a module-level logger with the standard pattern::

    import logging
    logger = logging.getLogger(__name__)

Call ``setup_logging`` once at the application entry point (e.g. in
``parallel_convolution.main``) to configure levels, format, and optional
warning capture::

    from .logger import setup_logging
    setup_logging(verbose=args.verbose, warnings_file=warnings_file)

After that call all module loggers automatically inherit the root configuration.
``verbose=True`` sets the root level to DEBUG (shows all ``logger.debug`` calls);
``verbose=False`` (the default) sets it to INFO.
"""

import logging


LOG_FORMAT = '[%(asctime)s] %(levelname)s %(name)s: %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'


def setup_logging(verbose=False, warnings_file=None):
    """Configure the root logger for the qsoabsfind application.

    Parameters
    ----------
    verbose : bool
        When *True* the root level is set to ``DEBUG`` so that all
        ``logger.debug(...)`` calls in every module become visible.
        When *False* (default) the level is ``INFO``.
    warnings_file : str or None
        If provided, Python ``warnings`` are captured via
        ``logging.captureWarnings`` and written to this file path in
        addition to the normal log stream.  Pass ``None`` to skip
        warning capture.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
    )

    if warnings_file is not None:
        import warnings
        logging.captureWarnings(True)
        warn_logger = logging.getLogger('py.warnings')
        warn_handler = logging.FileHandler(warnings_file)
        warn_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        warn_logger.addHandler(warn_handler)
        warn_logger.propagate = False
        # Suppress duplicate warnings on PyWarnings propagation
        warnings.simplefilter('always')
