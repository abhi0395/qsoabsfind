.. _qsoabsfind:

qsoabsfind's documentation
===========================

`qsoabsfind` is a Python module designed to detect absorbers with doublet properties in **SDSS** and **DESI** like low-resolution quasar spectra. It identifies potential absorption systems using a convolution-based, adaptive signal-to-noise approach, followed by Gaussian fitting and a series of rigorous checks to eliminate false positives.

The module also calculates rest-frame equivalent widths (EWs), FWHM and line centers using a double-Gaussian model. Optionally, it can calculate the total column densities of metal absorbers using the apparent optical depth method (AODM). The code offers flexibility to run with either default search parameters or user-provided custom search parameters.

Supported Metal Doublet Systems
-------------------------------

.. list-table::
   :widths: 15 20 20
   :header-rows: 1

   * - Absorber
     - Line 1 (Å)
     - Line 2 (Å)
   * - Mg II (Mg⁺)
     - 2796.35
     - 2803.52
   * - C IV (C³⁺)
     - 1548.20
     - 1550.77
   * - O VI (O⁵⁺)
     - 1031.93
     - 1037.62
   * - N V (N⁴⁺)
     - 1238.82
     - 1242.80
   * - Si IV (Si³⁺)
     - 1393.76
     - 1402.77
   * - Al III (Al²⁺)
     - 1854.72
     - 1862.79
   * - Fe II (Fe⁺)
     - 2586.65
     - 2600.17

Key Features
--------
- **Automated Search Window**: The code can dynamically define the observed-frame wavelength search window for each absorber system. Detailed definitions are provided in the `Search Window Documentation <https://qsoabsfind.readthedocs.io/en/latest/searchwindows.html>`_. Additionally, user can also provide the wavelength boundaries to search for metal systems through the search parameter config file.
- **Flexible Search Parameters:** Supports both default settings and user-provided custom search parameters for metal absorber detection.
- **Adaptive S/N convolution**: Detects doublet absorbers in low-resolution quasar spectra using a convolution-based, adaptive signal-to-noise method.
- **Rigorous selection criteria**: Identifies the best absorber candidates based on physically motivated thresholds and doublet properties. Optionally uses chi2 statistics to get the confidence level of the selected candidates.
- **Gaussian profile fitting**: Accurately models absorption lines to extract parameters like equivalent width, FWHM, and central wavelength.
- **Instrumental resolution correction**: Corrects measured line widths for instrumental resolution to infer intrinsic properties.
- **Column Densities**: Optionally estimates total column densities of detected absorbers using the apparent optical depth method (AODM; `Savage & Sembach 1991 <https://ui.adsabs.harvard.edu/abs/1991ApJ...379..245S/abstract>`_).
- **Parallel processing**: Supports efficient computation across large datasets using Python's ``multiprocessing`` module.
- **Descriptive Verbose**: Optionally prints the steps in great detail for debugging.


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


Contribution
------------

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas. If you have any questions/suggestions, please feel free to write to `abhijeetanand2011@gmail.com <mailto:abhijeetanand2011@gmail.com>`_ or, preferably, open a GitHub issue on the code `repo <https://github.com/abhi0395/qsoabsfind>`_.


| Thanks,
| Abhijeet Anand
| Lawrence Berkeley National Lab


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
