Running examples
================

Before running, please read :doc:`File formats <fileformat>`.

**1. Your input spectra** (absorber search without column densities):
----------------------------------------------------------------------

.. code-block:: bash

    qsoabsfind --input-fits-file $input \
               --constant-file $your_constant \
               --absorber $your_absorber \
               --output $output \
               --headers SURVEY=$YOUR_SURVEY AUTHOR=$YOUR_NAME \
               --ncpus 4

**2. Your input spectra** (absorber search with column densities):
-------------------------------------------------------------------

.. code-block:: bash

    qsoabsfind --input-fits-file $input \
               --constant-file $your_constant \
               --absorber $your_absorber \
               --output $output \
               --headers SURVEY=$YOUR_SURVEY AUTHOR=$YOUR_NAME \
               --ncpus 4 \
               --coldens-dv 300


Running with a YAML config file
-------------------------------

Instead of typing all arguments on the command line you can store them in a
YAML file and pass it with ``--config``. CLI flags always override YAML values.

.. code-block:: bash

    qsoabsfind --config example_config.yaml

    # Override individual values without editing the file:
    qsoabsfind --config example_config.yaml --absorber CIV --verbose

A fully annotated template is provided in ``data/example_config.yaml``.

.. note::

    YAML keys use underscores, not dashes (e.g. ``input_fits_file`` for
    ``--input-fits-file``). Set optional keys to ``null`` to let argparse
    use its built-in default.


Ready to run examples
---------------------

I have provided an example QSO spectra file, ``data/sdss/qso_test_spectra.fits``, which contains 100 continuum-normalized SDSS QSO spectra. You can use this file to test an example run as described below.

1. Without Column Densities
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**SDSS DR16 spectra** (MgII search)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

I have provided an example QSO spectra file, ``data/sdss/qso_test_spectra.fits``, which contains 100 continuum-normalized SDSS QSO spectra. These folders also have their own constants files. You can use these files to test example runs as described below.

.. code-block:: bash

    qsoabsfind --input-fits-file data/sdss/qso_test_spectra.fits \
               --constant-file data/sdss/sdss_constants.py \
               --absorber MgII \
               --output test_MgII.fits \
               --headers SURVEY=SDSS AUTHOR=YOUR_NAME \
               --ncpus 4

**DESI DR1 spectra** (CIV search):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Similarly, I have also provided an example QSO spectra file, ``data/desi/qso_test_spectra.fits``, which contains 100 continuum-normalized `DESI DR1 <https://data.desi.lbl.gov/doc/releases/dr1/>`_ QSO spectra. You can run absorber search on them as well.

.. code-block:: bash

    qsoabsfind --input-fits-file data/desi/qso_test_spectra.fits \
               --constant-file data/desi/desi_constants.py \
               --absorber CIV \
               --output test_CIV.fits \
               --headers SURVEY=DESI AUTHOR=YOUR_NAME \
               --ncpus 4

2. With Column Densities
^^^^^^^^^^^^^^^^^^^^^^^^

**SDSS DR16 spectra** (MgII search):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Optionally, users can instruct the module to calculate **total column densities** of metal absorbers using the
**apparent optical depth (AOD) method** (see `Savage & Sembach 1991 <https://ui.adsabs.harvard.edu/abs/1991ApJ...379..245S/abstract>`_).

To enable this feature, use the ``--coldens`` flag. You can also specify the velocity range using ``--dv``, which defines the maximum velocity (in km/s) on each side of the line center for integrating the optical depth.

Here, ``--dv 300`` means the integration will be performed over +/-300 km/s from each line center. You can adjust this value depending on the expected velocity width of the absorption lines.

.. code-block:: bash

    qsoabsfind --input-fits-file data/sdss/qso_test_spectra.fits \
               --constant-file data/sdss/desi_constants.py \
               --absorber MgII \
               --output test_MgII.fits \
               --headers SURVEY=SDSS AUTHOR=YOUR_NAME \
               --ncpus 4 \
               --coldens-dv 300

CLI Arguments
-------------

*Required* (may be supplied via ``--config`` instead of the command line):

.. option:: --input-fits-file <path>

   Path to the input FITS file containing continuum-normalised QSO spectra.

.. option:: --absorber <name>

   Doublet to search for. Choices: ``MgII``, ``CIV``, ``OVI``, ``NV``, ``SiIV``, ``AlIII``, ``FeII``, ``CaII``, ``NaI``.

.. option:: --constant-file <path>

   Path to the search-parameter constants ``.py`` file. See ``data/sdss/sdss_constants.py`` for the required format.

.. option:: --output <path>

   Output FITS catalog file path.

*Optional:*

.. option:: --config <path>

   Path to a YAML config file. Keys use underscores (e.g. ``input_fits_file``). CLI flags always
   override YAML values. A fully annotated template is provided at ``data/example_config.yaml``.

.. option:: --n-qso <value>

   Which spectra to process. Accepts a single integer (``500``), a range (``1-1000``), or a
   stepped range (``1-1000:10``). Default: all spectra in the file.

.. option:: --headers <NAME=VALUE> [<NAME=VALUE> ...]

   One or more ``KEY=VALUE`` pairs written as keywords to the output FITS PRIMARY HDU
   (e.g. ``SURVEY=SDSS AUTHOR=YOUR_NAME``).

.. option:: --ncpus <int>

   Number of parallel worker processes. Default: ``4``.

.. option:: --coldens-dv <float>

   If provided, also computes total column densities for each detected absorber using the apparent
   optical depth method (AODM; Savage & Sembach 1991). The value sets the +/- velocity range (km/s)
   for optical-depth integration around each line centre (e.g. ``300``). Adds a ``COLUMN_DENSITY``
   HDU to the output file.

.. option:: --verbose

   Enable detailed per-spectrum and debug logging to the terminal and log file.

.. option:: --zabs-known-file <path>

   Path to a FITS file with columns ``INDEX_SPEC`` and ``Z_ABS``. When supplied, the convolution
   search is skipped and Gaussian fitting is run at the provided redshifts only. Multiple rows
   with the same ``INDEX_SPEC`` are treated as multiple known redshifts for that spectrum. Adds
   a ``ZABS_KNOWN`` column in the ``ABSORBER`` HDU.

   The ``max_dv_known`` key in the constants file controls how far (in km/s) the fitted redshift
   is allowed to drift from the seed before the detection is rejected (default: 500 km/s).


Useful notes
-------------

Parallel mode can be memory-intensive if the input FITS file is large in size. As the code accesses the FITS file to read QSO spectra when running in parallel, it can become a bottleneck for memory, and the code may fail. Currently, I suggest the following:

   - **Divide your file into smaller chunks:** Split the FITS file into several smaller files, each containing approximately `N` spectra. Then run the code on these smaller files.

   - **Use a rule of thumb for file size:** Ensure that the size of each individual file is no larger than `total_memory/ncpu` of your node or system. Based on this idea you can decide your `N`. I would suggest `N = 1000`.

   - **Merge results at the end:** After processing, you can merge your results using `qsoabsfind.utils.combine_fits_files <https://github.com/abhi0395/qsoabsfind/blob/main/qsoabsfind/utils.py>`_ function. Please read the description before using it.

In order to decide the right size of the FITS file, consider the total available memory and the number of CPUs in your system.
