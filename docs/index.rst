.. _qsoabsfind:

qsoabsfind's documentation
===========================

`qsoabsfind` is a Python module designed to detect absorbers with doublet properties in SDSS/DESI quasar spectra. This tool identifies potential absorbers using a convolution-based adaptive S/N approach, applies Gaussian fitting and extensive checks to reject false positives, and computes equivalent widths (EWs) of the lines using a simple double Gaussian.

Currently, the package only works for **MgII 2796,2803** and **CIV 1548,1550** doublets.

Features
--------

- Convolution-based adaptive S/N approach for detecting absorbers in QSO spectra.
- Gaussian fitting for accurate measurement of absorber properties (such as EW, line widths, and centers).
- Parallel processing for efficient computation on a large number of spectra.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation
   fileformat
   qsoabsfind


Citation
--------

Please cite `Anand, Nelson & Kauffmann 2021 <https://arxiv.org/abs/2103.15842>`_ if you find this code useful in your research. The BibTeX entry for the paper is:

.. code-block:: sh

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

Contact
-------

Abhijeet Anand  \
Lawrence Berkeley National Lab  

If you have any questions/suggestions, please feel free to write to abhijeetanand2011 [at] gmail.com


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
