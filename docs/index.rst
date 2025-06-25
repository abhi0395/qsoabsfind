.. _qsoabsfind:

qsoabsfind's documentation
===========================

``qsoabsfind`` is a Python module designed to detect absorbers with doublet properties in **SDSS** and **DESI** quasar spectra. It identifies potential absorption systems using a convolution-based, adaptive signal-to-noise approach, followed by Gaussian fitting and a series of rigorous checks to eliminate false positives. The module also calculates rest-frame equivalent widths (EWs), FWHM and line centers using a double-Gaussian model.

Currently, the package supports only **Mg II (2796, 2803 Å)** and **C IV (1548, 1550 Å)** doublets.

Features
--------

- **Adaptive S/N convolution**: Detects doublet absorbers in low-resolution quasar spectra using a convolution-based, adaptive signal-to-noise method.
- **Rigorous selection criteria**: Identifies the best absorber candidates based on physically motivated thresholds and doublet properties.
- **Gaussian profile fitting**: Accurately models absorption lines to extract parameters like equivalent width, FWHM, and central wavelength.
- **Instrumental correction**: Corrects measured line widths for instrumental resolution to infer intrinsic properties.
- **Parallel processing**: Supports efficient computation across large datasets using Python's ``multiprocessing`` module.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   fileformat
   qsoabsfind


GitHub repository
-----------------

The source code is available on GitHub. Please see the `qsoabsfind <https://github.com/abhi0395/qsoabsfind>`_ repo. An `example jupyter notebook <https://github.com/abhi0395/qsoabsfind/blob/main/nb>`_ is also available.


Citation
--------

If you use this code in your analysis, please cite `Anand, Nelson & Kauffmann 2021 <https://arxiv.org/abs/2103.15842>`_ and `Anand et al. 2025 <https://arxiv.org/abs/2504.20299>`_. The BibTeX entries for these papers can be found `here (2021 paper) <https://ui.adsabs.harvard.edu/abs/2021MNRAS.504...65A/exportcitation>`_ and `here (2025 paper) <https://ui.adsabs.harvard.edu/abs/2025arXiv250420299A/exportcitation>`_.

If you use this **codebase**, please also cite the associated `Zenodo record <https://zenodo.org/records/15685771>`_. Additionally, consider starring the repository if you find it useful or use it in your work.


Contact
-------

| Abhijeet Anand
| Lawrence Berkeley National Lab
|

For questions or feedback, please contact: `abhijeetanand2011@gmail.com <mailto:abhijeetanand2011@gmail.com>`_ or, preferably, open a GitHub issue on the code `repo <https://github.com/abhi0395/qsoabsfind>`_.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
