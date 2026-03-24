.. _qsoabsfind:

.. title:: qsoabsfind
.. image:: images/logo.png
   :width: 450px
   :height: 305px
   :align: center

|

qsoabsfind: Quasar Absorber Finder
----------------------------------

`qsoabsfind` is a Python module designed to detect absorbers with doublet properties (absorbers with two close lines) in low-resolution quasar spectra (e.g. SDSS, DESI, MUSE, 4MOST, WAVES, WEAVE etc.). It identifies potential absorption systems using a convolution-based, adaptive signal-to-noise approach, followed by Gaussian fitting and a series of rigorous checks to eliminate false positives.

The module also calculates rest-frame equivalent widths (EWs), FWHM and line centers using a double-Gaussian model. Optionally, it can calculate the total column densities of metal absorbers using the apparent optical depth method (AODM). The code offers flexibility to run with either default search parameters or user-provided custom search parameters.

Supported Metal Doublet Systems
-------------------------------

.. list-table::
   :widths: 15 20 20
   :header-rows: 1

   * - Absorber
     - Line 1 (Ang)
     - Line 2 (Ang)
   * - O VI (O⁵⁺)
     - 1031.93
     - 1037.62
   * - N V (N⁴⁺)
     - 1238.82
     - 1242.80
   * - Si IV (Si³⁺)
     - 1393.76
     - 1402.77
   * - C IV (C³⁺)
     - 1548.20
     - 1550.77
   * - Al III (Al²⁺)
     - 1854.72
     - 1862.79
   * - Fe II (Fe⁺)
     - 2586.65
     - 2600.17
   * - Mg II (Mg⁺)
     - 2796.35
     - 2803.52
   * - Ca II (Ca⁺)
     - 3934.78
     - 3969.59
   * - Na I (Na)
     - 5891.58
     - 5897.57


Key Features
------------

- **Automated Search Window**: The code can dynamically define the observed-frame wavelength search window for each absorber system. Detailed definitions are provided in the `Search Window Documentation <https://qsoabsfind.readthedocs.io/en/latest/searchwindows.html>`_. Additionally, user can also provide the wavelength boundaries to search for metal systems through the search parameter constants file.
- **9 built-in doublet systems**: Automatic search-window calculation and line properties are pre-configured for MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, and NaI.
- **Extensible to any doublet**: The pipeline is generic — supply a custom constants file with your doublet's rest-frame wavelengths, oscillator strengths, and search bounds (see ``Parameters File``), and the pipeline will search for it. Custom systems are functional but not as thoroughly tested as the built-ins.
- **Adaptive S/N convolution**: Detects doublet absorbers in low-resolution quasar spectra using a convolution-based, adaptive signal-to-noise method.
- **Rigorous selection criteria**: Identifies the best absorber candidates based on physically motivated thresholds and doublet properties. Optionally uses chi2 statistics to get the confidence level of the selected candidates.
- **Gaussian profile fitting**: Accurately models absorption lines to extract parameters like equivalent width, FWHM, and central wavelength.
- **Instrumental resolution correction**: Corrects measured line widths for instrumental resolution to infer intrinsic properties.
- **Known-redshift validation**: When a prior absorber catalog (e.g. from another survey or absorber finder or catalog built from ``qsoabsfind``) is available, ``--zabs-known-file`` skips the convolution search for the given absorbers and runs Gaussian fitting and selection only at the supplied redshifts, enabling fast validation of known systems.
- **Column Densities**: Optionally estimates total column densities of detected absorbers using the apparent optical depth method (AODM; `Savage & Sembach 1991 <https://ui.adsabs.harvard.edu/abs/1991ApJ...379..245S/abstract>`_).
- **Parallel processing**: Supports efficient computation across large datasets using Python's ``multiprocessing`` module.
- **Comprehensive Output**: Detailed catalogs with redshifts, equivalent widths, S/N ratios, and more.
- **Descriptive Verbose**: Optionally prints the steps in great detail for debugging.
- **Visualization**: Plot the full spectrum with all detected absorber systems marked, plus zoomed panels around each detection, using ``plot_multiple_metal_systems``.

**qsoabsfind** is suitable for
------------------------------
- Large absorber catalog construction
- Metal-line evolution studies
- CGM/IGM absorber statistics
- Survey-scale quasar spectral analysis

Pre-filtering searchable QSOs
------------------------------

Before running the full absorber search you can quickly flag which spectra
have a usable wavelength window for the absorber of interest.
``qsoabsfind.absorberutils.find_searchable_qsos`` performs this check
in parallel over the whole file and returns a two-column
``astropy.table.Table`` (``QSO_INDEX``, ``IS_GOOD``) that can be used to
build a parent sample:

.. code-block:: python

    from qsoabsfind.absorberutils import find_searchable_qsos

    parent = find_searchable_qsos(
        fits_file='spectra.fits',
        absorber='MgII',
        constant_file='my_constants.py',
        ncpus=8,        # parallel workers
        n_qso=None,     # None = all spectra; or '1-5000', '500', '1-5000:2'
        verbose=False,
    )

    # keep only searchable QSOs
    good = parent[parent['IS_GOOD']]
    print(f"{len(good)} / {len(parent)} QSOs have a searchable MgII window")

The function applies the same overridable-constants logic as the
main pipeline, so the filtering is fully consistent with the absorber search.


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

Reading output catalogs
-----------------------

After running the absorber search, you can load the output FITS catalog using the ``AbsorberData`` class:

.. code-block:: python

      from qsoabsfind.datamodel import AbsorberData

      catalog = AbsorberData('test_MgII.fits', autoload=True)

      print(catalog.catalog)        # absorber table (ABSORBER HDU)
      print(catalog.metadata)       # QSO metadata for detected absorbers (METADATA HDU)
      print(catalog.qso_info)       # all processed spectra with IS_QSO_AVAILABLE flag (QSO_INFO HDU)
      print(catalog.column_density) # column densities if --coldens was used, else None


Plotting a random absorber
--------------------------

Once you have loaded the spectra and the output catalog, you can visualise a
randomly selected absorber using :func:`qsoabsfind.utils.plot_absorber`:

.. code-block:: python

    import numpy as np
    from qsoabsfind.datamodel import QSOSpecRead, AbsorberData
    from qsoabsfind.utils import plot_absorber

    # Load the output absorber catalog
    catalog = AbsorberData('/path/to/your/absorber.fits', autoload=True)

    # Pick a random absorber from the catalog
    rng = np.random.default_rng()
    idx = rng.integers(len(catalog.catalog))
    row = catalog.catalog[idx]

    # Load the corresponding QSO spectrum
    spectra = QSOSpecRead('/path/to/your/spectra.fits',
                          index=int(row['INDEX_SPEC']),
                          autoload=True)

    # Plot the absorber (full spectrum + zoomed-in doublet view)
    plot_absorber(spectra, absorber='MgII', zabs=row,
                  title=f"MgII absorber at z={row['Z_ABS']:.4f}")

Pass ``show_error=True`` to overlay the error spectrum, or
``plot_filename='absorber.png'`` to save the figure to disk instead of
displaying it interactively.

Plotting all absorbers across multiple systems
----------------------------------------------

To visualise every detected system in a given spectrum, use
:func:`qsoabsfind.utils.plot_multiple_metal_systems`. It plots the full
spectrum with all systems annotated, followed by a zoomed panel for each
individual detection. It assumes that both catalogs are one to one mapped
to the same input spectra (e.g. both catalogs were generated from the same
input file) and uses the ``INDEX_SPEC`` column to match absorbers across
different systems:

.. code-block:: python

   from qsoabsfind.io import QSOSpecRead
   from qsoabsfind.utils import plot_multiple_metal_systems
   from astropy.io import fits
   from astropy.table import Table

   # Load spectrum
   spectra = QSOSpecRead('spectra.fits', qso_index=0)

   # Load catalogs for two absorbers
   with fits.open('output_MgII.fits') as hdul:
       mgii = Table(hdul['ABSORBER'].data)
   with fits.open('output_CIV.fits') as hdul:
       civ = Table(hdul['ABSORBER'].data)

   # Plot full spectrum + zoomed panels for every detected system
   plot_multiple_metal_systems(
       spectra,
       absorber_dict={'MgII': mgii, 'CIV': civ},
       zoom=True,
       plot_filename='absorbers.pdf',   # or None to display interactively
   )


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   fileformat
   searchwindows
   paramfile
   examplerun
   qsoabsfind


GitHub repository
-----------------

The source code is available on GitHub. Please see the `qsoabsfind <https://github.com/abhi0395/qsoabsfind>`__ repo. An `example jupyter notebook <https://github.com/abhi0395/qsoabsfind/blob/main/nb>`_ is also available.


Citation
--------

If you use this code in your analysis, please cite `Anand, Nelson & Kauffmann 2021 <https://arxiv.org/abs/2103.15842>`_ and `Anand et al. 2025 <https://arxiv.org/abs/2504.20299>`_. The BibTeX entries for these papers can be found below. Otherwise, they can also be copied from `here (2021 paper) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.504...65A/exportcitation>`_ and `here (2025 paper) <https://ui.adsabs.harvard.edu/abs/2025arXiv250420299A/exportcitation>`_.

If you use this **codebase**, please also cite the associated `Zenodo record <https://zenodo.org/records/15685771>`_. Additionally, consider starring the repository if you find it useful or use it in your work.

You can also copy the BibTeX entry directly from below.

.. code-block:: bibtex

    @ARTICLE{2021MNRAS.504...65A,
      author = {{Anand}, Abhijeet and {Nelson}, Dylan and {Kauffmann}, Guinevere},
      title = "{Characterizing the abundance, properties, and kinematics of the cool circumgalactic medium of galaxies in absorption with SDSS DR16}",
      journal = {\mnras},
      keywords = {galaxies: evolution, galaxies: formation, large-scale structure of Universe, Astrophysics - Astrophysics of Galaxies},
      year = 2021,
      month = jun,
      volume = {504},
      number = {1},
      pages = {65-88},
      doi = {10.1093/mnras/stab871},
      archivePrefix = {arXiv},
      eprint = {2103.15842},
      primaryClass = {astro-ph.GA},
      adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.504...65A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @ARTICLE{2025ApJ...990..151A,
      author = {{Anand}, Abhijeet and {Aguilar}, J. and {Ahlen}, S. and {Bianchi}, D. and {Brodzeller}, A. and {Brooks}, D. and {Canning}, R. and {Claybaugh}, T. and {Cuceu}, A. and {de la Macorra}, A. and {Doel}, P. and {Ferraro}, S. and {Font-Ribera}, A. and {Forero-Romero}, J.~E. and {Gazta{\~n}aga}, E. and {Gontcho A Gontcho}, S. and {Gutierrez}, G. and {Guy}, J. and {Herrera-Alcantar}, H.~K. and {Ishak}, M. and {Juneau}, S. and {Kehoe}, R. and {Kremin}, A. and {Landriau}, M. and {Le Guillou}, L. and {Levi}, M.~E. and {Manera}, M. and {Meisner}, A. and {Miquel}, R. and {Moustakas}, J. and {Mu{\~n}oz-Guti{\'e}rrez}, A. and {Napolitano}, L. and {P{\'e}rez-R{\`a}fols}, I. and {Rossi}, G. and {Sanchez}, E. and {Schlegel}, D. and {Schubnell}, M. and {Sprayberry}, D. and {Tarl{\'e}}, G. and {Temple}, M.~J. and {Weaver}, B.~A. and {Zhou}, R.},
      title = "{The Cosmic Evolution of C IV Absorbers at 1.4 < z < 4.5: Insights from 100,000 Systems in DESI Quasars}",
      journal = {\apj},
      keywords = {Quasar absorption line spectroscopy, Intergalactic medium, Redshift surveys, Astronomy software, 1317, 813, 1378, 1855, Cosmology and Nongalactic Astrophysics},
      year = 2025,
      month = sep,
      volume = {990},
      number = {2},
      eid = {151},
      pages = {151},
      doi = {10.3847/1538-4357/adef3c},
      archivePrefix = {arXiv},
      eprint = {2504.20299},
      primaryClass = {astro-ph.CO},
      adsurl = {https://ui.adsabs.harvard.edu/abs/2025ApJ...990..151A},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
    }

    @software{Anandqsoabsfind2025,
      author    = {{Anand}, Abhijeet},
      title     = "{qsoabsfind: A Python Package for Detecting Absorption Line Doublets in SDSS and DESI Quasar Spectra}",
      month     = jun,
      year      = 2025,
      publisher = {Zenodo},
      doi       = {10.5281/zenodo.15685771},
      url       = {https://doi.org/10.5281/zenodo.15685771}
    }


Contribution
------------

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas. If you have any questions/suggestions, please feel free to write to `abhijeetanand2011@gmail.com <mailto:abhijeetanand2011@gmail.com>`_ or, preferably, open a GitHub issue on the code `repo <https://github.com/abhi0395/qsoabsfind>`_.

Acknowledgements
----------------

The first crude version of the code was developed and written by me during my PhD with lots of suggestions from my PhD supervisors `Prof. Dr. Guinevere Kauffmann <https://www.mpa-garching.mpg.de/person/44092>`_ and `Dr. Dylan Nelson <https://nelson.tng-project.org>`_. Over the years, it has evolved from a specialized script into the generic, community-ready framework it is today. I would like to extend my thanks to the VS Code AI agents, which were instrumental in refining the codebase. They provided invaluable assistance in documenting functions, optimizing logic, and expanding unit test coverage. They helped ensure the code is both robust and maintainable. The project logo was created from a absorber example generated by me, with assistance from ChatGPT-5.


| Thanks,
| Abhijeet Anand
| IUCAA, Pune & Lawrence Berkeley National Lab


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
