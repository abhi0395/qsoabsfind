.. _qsoabsfind:

qsoabsfind's documentation
===========================

`qsoabsfind` is a Python module designed to detect absorbers with doublet properties in SDSS/DESI quasar spectra. This tool identifies potential absorbers using a convolution-based adaptive S/N approach, applies Gaussian fitting and extensive checks to reject false positives, and computes equivalent widths (EWs) of the lines using a simple double Gaussian.

Currently, the package only works for **MgII 2796,2803** and **CIV 1548,1550** doublets.

Features
--------

- Convolution-based adaptive S/N approach for detecting doublet absorbers in low-resolution QSO spectra.
- A rigorous selection criterion to select the best absorber candidates.
- Gaussian fitting for accurately measuring absorber properties (such as EW, line widths, and centers).
- Instrumental-resolution correction to the width of detected absorbers.
- Parallel processing using `multiprocessing` for efficient computation on a large number of spectra.

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


Contact
-------

| Abhijeet Anand
| Lawrence Berkeley National Lab

If you have any questions/suggestions, please feel free to write to abhijeetanand2011 [at] gmail.com or, preferably, open a GitHub issue.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
