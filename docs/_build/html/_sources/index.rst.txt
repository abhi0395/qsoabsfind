.. _qsoabsfind:

qsoabsfind's documentation
===========================

`qsoabsfind` is a Python module designed to detect absorbers with doublet properties in **SDSS** and **DESI** like low-resolution quasar spectra. It identifies potential absorption systems using a convolution-based, adaptive signal-to-noise approach, followed by Gaussian fitting and a series of rigorous checks to eliminate false positives. The module also calculates rest-frame equivalent widths (EWs), FWHM and line centers using a double-Gaussian model. Optionally, it can calculate the total column densities of metal absorbers using the apparent optical depth method (AODM). The code offers flexibility to run with either default search parameters or user-provided custom search parameters.

Supported Metal Doublet Systems
-------------------------------

.. list-table::
   :widths: 15 12 20 12 20
   :header-rows: 1

   * - Absorber
     - Line 1 (Å)
     - Oscillator Strength 1
     - Line 2 (Å)
     - Oscillator Strength 2
   * - Mg II (Mg⁺)
     - 2796.35
     - 0.6123
     - 2803.52
     - 0.3054
   * - C IV (C³⁺)
     - 1548.20
     - 0.1900
     - 1550.77
     - 0.0952
   * - O VI (O⁵⁺)
     - 1031.93
     - 0.1329
     - 1037.62
     - 0.0661
   * - N V (N⁴⁺)
     - 1238.82
     - 0.1570
     - 1242.80
     - 0.0782
   * - Si IV (Si³⁺)
     - 1393.76
     - 0.5140
     - 1402.77
     - 0.2553
   * - Al III (Al²⁺)
     - 1854.72
     - 0.5390
     - 1862.79
     - 0.2680
   * - Fe II (Fe⁺)
     - 2586.65
     - 0.0691
     - 2600.17
     - 0.2394

Key Features
--------
- **Automated Search Window**: The code dynamically defines the observed-frame wavelength search window for each absorber system. Detailed definitions are provided in the `Search Window Documentation <https://qsoabsfind.readthedocs.io/en/latest/searchwindows.html>`_.
- **Adaptive S/N convolution**: Detects doublet absorbers in low-resolution quasar spectra using a convolution-based, adaptive signal-to-noise method.
- **Rigorous selection criteria**: Identifies the best absorber candidates based on physically motivated thresholds and doublet properties.
- **Gaussian profile fitting**: Accurately models absorption lines to extract parameters like equivalent width, FWHM, and central wavelength.
- **Instrumental correction**: Corrects measured line widths for instrumental resolution to infer intrinsic properties.
- **Column Densities**: Optionally estimates total column densities of detected absorbers using the apparent optical depth method (AODM; `Savage & Sembach 1991 <https://ui.adsabs.harvard.edu/abs/1991ApJ...379..245S/abstract>`_).
- **Flexible Search Parameters:** Supports both default settings and user-provided custom search parameters for metal absorber detection.
- **Parallel processing**: Supports efficient computation across large datasets using Python's ``multiprocessing`` module.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   fileformat
   searchwindows
   examplerun
   qsoabsfind


GitHub repository
-----------------

The source code is available on GitHub. Please see the `qsoabsfind <https://github.com/abhi0395/qsoabsfind>`_ repo. An `example jupyter notebook <https://github.com/abhi0395/qsoabsfind/blob/main/nb>`_ is also available.


Citation
--------

If you use this code in your analysis, please cite `Anand, Nelson & Kauffmann 2021 <https://arxiv.org/abs/2103.15842>`_ and `Anand et al. 2025 <https://arxiv.org/abs/2504.20299>`_. The BibTeX entries for these papers can be found `here (2021 paper) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.504...65A/exportcitation>`_ and `here (2025 paper) <https://ui.adsabs.harvard.edu/abs/2025arXiv250420299A/exportcitation>`_.

If you use this **codebase**, please also cite the associated `Zenodo record <https://zenodo.org/records/15685771>`_. Additionally, consider starring the repository if you find it useful or use it in your work.

Contact & Issues
---------------

For questions or feedback, please contact: `abhijeetanand2011@gmail.com <mailto:abhijeetanand2011@gmail.com>`_ or, preferably, open a GitHub issue on the code `repo <https://github.com/abhi0395/qsoabsfind>`_.


| Thanks,
| Abhijeet Anand
| Lawrence Berkeley National Lab


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
