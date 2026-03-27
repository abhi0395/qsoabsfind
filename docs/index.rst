.. _qsoabsfind:

.. title:: qsoabsfind
.. image:: images/logo.png
   :width: 450px
   :height: 305px
   :align: center

|

qsoabsfind: Quasar Absorber Finder
----------------------------------

`qsoabsfind` is a Python module for detecting absorbers with doublet properties (absorbers with two close lines) in low-resolution quasar spectra (e.g. SDSS, DESI, MUSE, 4MOST, WAVES, WEAVE etc.). It identifies absorption systems using a convolution-based, adaptive signal-to-noise approach, followed by Gaussian fitting and a set of selection checks to eliminate false positives.

The module also calculates rest-frame equivalent widths (EWs), FWHM and line centers using a double-Gaussian model or trapezoidal integration method. Optionally, it can calculate the total column densities of metal absorbers using the apparent optical depth method (AODM). It can run with either the default search parameters or user-provided custom ones.

It also provides the ability to search for additional absorber systems at the redshifts of known absorbers. This is useful for constructing multi-line absorber catalogs and for validating detections using multiple transitions.

Default Metal Doublet Systems
-----------------------------

.. list-table::
   :widths: 15 20 20 20
   :header-rows: 1

   * - Absorber
     - Line 1 (Ang)
     - Line 2 (Ang)
     - Comments
   * - O VI (O⁵⁺)
     - 1031.93
     - 1037.62
     - Lines fall inside the Ly-alpha forest; makes detection and confirmation difficult
   * - N V (N⁴⁺)
     - 1238.82
     - 1242.80
     - Lies near the red edge of the Ly-alpha forest; avoiding the forest leaves a very short absorber path length, but detection is feasible
   * - Si IV (Si³⁺)
     - 1393.76
     - 1402.77
     - Outside the Ly-alpha forest; relatively clean spectral region, easier to detect and confirm
   * - C IV (C³⁺)
     - 1548.20
     - 1550.77
     - Outside the Ly-alpha forest; one of the strongest UV doublets, easy to detect and confirm
   * - Al III (Al²⁺)
     - 1854.72
     - 1862.79
     - Outside the Ly-alpha forest; clean region
   * - Fe II (Fe⁺)
     - 2586.65
     - 2600.17
     - Outside the Ly-alpha forest; clean region; large line separation, easy to detect and confirm
   * - Mg II (Mg⁺)
     - 2796.35
     - 2803.52
     - Outside the Ly-alpha forest; large line separation, easy to detect and confirm
   * - Ca II (Ca⁺)
     - 3934.78
     - 3969.59
     - Outside the Ly-alpha forest; though can lie in sky line region, which may make it difficult
   * - Na I (Na⁰)
     - 5891.58
     - 5897.57
     - Outside the Ly-alpha forest; though can lie in sky line region, which may make it difficult

Note on Absorbers
-----------------

The pipeline is generic. Users can supply a custom constants file with your doublet's rest-frame wavelengths, oscillator strengths, and search bounds (see :doc:`Parameter File <paramfile>`), and the pipeline will search for it. Custom systems are functional but not as *thoroughly tested as the default ones*.


Key Features
------------

- **Automated Search Window**: The code can dynamically define the observed-frame wavelength search window for each absorber system. Detailed definitions are provided in the `Search Window Documentation <https://qsoabsfind.readthedocs.io/en/latest/searchwindows.html>`_. Additionally, user can also provide the wavelength boundaries to search for metal systems through the search parameter constants file.
- **9 built-in doublet systems**: Automatic search-window calculation and line properties are pre-configured for MgII, CIV, OVI, NV, SiIV, AlIII, FeII, CaII, and NaI. **OVI** is very hard as it lies in the Ly-alpha forest. So use with caution.
- **Extensible to any doublet**: The pipeline is generic. Users can supply a custom constants file with your doublet's rest-frame wavelengths, oscillator strengths, and search bounds (see ``Parameters File``), and the pipeline will search for it. Custom systems are functional but not as *thoroughly tested as the default ones*.
- **Adaptive S/N convolution**: Detects doublet absorbers in low-resolution quasar spectra using a convolution-based, adaptive signal-to-noise method.
- **Selection criteria**: Identifies absorber candidates based on S/N thresholds and doublet properties. Optionally uses chi2 statistics to get the confidence level of the selected candidates.
- **Gaussian profile fitting**: Fits absorption lines with a double-Gaussian model to extract equivalent width, FWHM, and central wavelength.
- **Instrumental resolution correction**: Corrects measured line widths for instrumental resolution to infer intrinsic properties.
- **Known-redshift validation**: When a prior absorber catalog (e.g. from another survey or absorber finder or catalog built from ``qsoabsfind``) is available, ``--zabs-known-file`` skips the convolution search and runs Gaussian fitting and selection only at the supplied redshifts, allowing quick validation of known systems.
- **Column Densities**: Optionally estimates total column densities of detected absorbers using the apparent optical depth method (AODM; `Savage & Sembach 1991 <https://ui.adsabs.harvard.edu/abs/1991ApJ...379..245S/abstract>`_). Can be turned on via ``--coldens-dv`` to specify the velocity range for integration.
- **Parallel processing**: Runs across large datasets using Python's ``multiprocessing`` module.
- **Detailed output**: Catalogs with redshifts, equivalent widths, S/N ratios, and more.
- **Verbose mode**: Optionally prints each processing step for debugging.
- **Trapezoidal EW measurement**: In addition to Gaussian-model EWs, computes rest-frame equivalent widths via direct trapezoidal integration (can be provided via ``--trapz-ew-sigma``) over a per-line window of :math:`\pm n \cdot \sigma` centred on each Gaussian-fit line centre. For close doublets (e.g. C IV with a 2.6 Ang separation), the integration windows are automatically clipped at the doublet midpoint to prevent double-counting. Measurement windows and integrated areas can be visualised with ``plot_trapezoidal_ew_windows``.
- **Visualization**: Plot the full spectrum with all detected absorber systems marked, plus zoomed panels around each detection, using ``plot_multiple_metal_systems``.

**qsoabsfind** is suitable for
------------------------------
- Large absorber catalog construction
- Metal-line evolution studies
- CGM/IGM absorber statistics
- Survey-scale quasar spectral analysis


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   fileformat
   searchwindows
   paramfile
   examplerun
   preprocessing
   postprocessing
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

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas or if you find any bugs. If you have any questions/suggestions, please feel free to write to `abhijeetanand2011@gmail.com <mailto:abhijeetanand2011@gmail.com>`_ or, preferably, open a GitHub issue on the code `repo <https://github.com/abhi0395/qsoabsfind>`_.

Acknowledgements
----------------

The first crude version of the code was developed and written by me during my PhD with lots of suggestions from my PhD supervisors `Prof. Dr. Guinevere Kauffmann <https://www.mpa-garching.mpg.de/person/44092>`_ and `Dr. Dylan Nelson <https://nelson.tng-project.org>`_. Over the years, it has grown from a simple script into a general-purpose tool. I thank the VS Code AI agents for their help in improving the codebase — they helped with documenting functions, optimizing logic, and expanding unit test coverage. The project logo was created from a absorber example generated by me, with assistance from ChatGPT-5.

Disclaimer
----------

Like any software, this code may contain bugs or unintended behavior. It is provided "as is" without warranty of any kind. Users are encouraged to test the code on a small sample before applying it to large datasets. If you find any issues, please report them via GitHub.

| Thanks,
| Abhijeet Anand
| IUCAA, Pune & Lawrence Berkeley National Lab


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
