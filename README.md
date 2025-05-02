qsoabsfind
============

**The Python module designed to detect absorbers with doublet properties in SDSS/DESI quasar**

[![github shields.io](https://img.shields.io/badge/GitHub-abhi0395%2Fqsoabsfind-blue.svg?style=flat)](https://github.com/abhi0395/qsoabsfind)
[![Tests](https://github.com/abhi0395/qsoabsfind/actions/workflows/tests.yml/badge.svg)](https://github.com/abhi0395/qsoabsfind/actions)
[![license shields.io](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/abhi0395/qsoabsfind/blob/main/LICENSE)
[![arXiv-2103.15842](http://img.shields.io/badge/arXiv-2103.15842-orange.svg?style=flat)](https://arxiv.org/abs/2103.15842)
[![Documentation Status](https://readthedocs.org/projects/qsoabsfind/badge/?version=latest)](https://qsoabsfind.readthedocs.io/en/latest/?badge=latest)

`qsoabsfind` is a Python module designed to detect absorbers with doublet properties in SDSS/DESI quasar spectra. This tool identifies potential absorbers using a convolution-based adaptive S/N approach, applies Gaussian fitting and extensive checks to reject false positives, and computes equivalent widths (EWs) of the lines using a simple double Gaussian.

Currently, the package only works for **MgII 2796,2803** and **CIV 1548,1550** doublets.

Features
--------

- Convolution-based adaptive S/N approach for detecting doublet absorbers in low-resolution QSO spectra.
- A rigorous selection criterion to select the best absorber candidates.
- Gaussian fitting for accurately measuring absorber properties (such as EW, line widths, and centers).
- Instrumental-resolution correction to the width of detected absorbers.
- Parallel processing using `multiprocessing` for efficient computation on a large number of spectra.

Documentation
-------------

The full documentation is available at [https://qsoabsfind.readthedocs.io](https://qsoabsfind.readthedocs.io).

Installation
------------

Prerequisites
-------------

- Python 3.6 or higher
- `numpy`
- `scipy`
- `astropy`
- `numba`
- `matplotlib`
- `pytest` (for running tests)

Clone the Repository
--------------------

First, clone the repository to your local machine:

```sh
git clone https://github.com/abhi0395/qsoabsfind.git
cd qsoabsfind
pip install .
python -m unittest discover -s tests

```

Instructions
-------------

- Before running the module, please read the [datamodel](https://github.com/abhi0395/qsoabsfind/blob/main/data/datamodel.rst). The instructions for the input and output files are provided there. 
- I have also provided two example QSO spectra files:
  -  `data/sdss/qso_test_spectra.fits` : 500 continuum-normalized spectra from [SDSS DR16](https://www.sdss4.org/dr17/algorithms/qso_catalog/) 
  -  `data/desi/qso_test_spectra.fits` : 500 continuum-normalized spectra from [DESI DR1](https://data.desi.lbl.gov/doc/releases/dr1/)
- You can use this file to test an example run as described below.

Running as script:
----------------

**SDSS DR16 Spectra**
---------------------

```sh
qsoabsfind --input-fits-file data/sdss/qso_test_spectra.fits \
           --n-qso 500 \
           --absorber MgII \
           --output test_MgII.fits \
           --headers SURVEY=SDSS AUTHOR=YOUR_NAME \
           --ncpus 4 \
           --constant-file data/sdss/sdss_constants.py
```

**DESI DR1 Spectra**
---------------------

```sh
qsoabsfind --input-fits-file data/desi/qso_test_spectra.fits \
           --n-qso 500 \
           --absorber MgII \
           --output test_MgII.fits \
           --headers SURVEY=DESI AUTHOR=YOUR_NAME \
           --ncpus 4 \
           --constant-file data/desi/desi_constants.py
```

Useful notes:
-------------

Parallel mode can be memory-intensive if the input FITS file is large in size. As the code accesses the FITS file to read QSO spectra when running in parallel, it can become a bottleneck for memory, and the code may fail. Currently, I suggest the following:

- **Divide your file into smaller chunks:** Split the FITS file into several smaller files, each containing approximately `N` spectra. Then run the code on these smaller files.

- **Use a rule of thumb for file size:** Ensure that the size of each individual file is no larger than `total_memory/ncpu` of your node or system. Based on this idea you can decide your `N`. I would suggest `N = 1000`.

- **Merge results at the end:** After processing, you can merge your results using `qsoabsfind.utils.combine_fits_files`.

In order to decide the right size of the FITS file, consider the total available memory and the number of CPUs in your system.

Example catalog runs
--------------------

SDSS and DESI [example jupyter notebooks](https://github.com/abhi0395/qsoabsfind/blob/main/nb/) are also available.

Contribution
------------

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas. If you have any questions/suggestions, please feel free to write to abhijeetanand2011@gmail.com or, preferably, open a GitHub issue.

Citation
--------

If you use this code in your analysis, please cite [Anand, Nelson & Kauffmann 2021](https://arxiv.org/abs/2103.15842) and [Anand et al. 2025](https://arxiv.org/abs/2504.20299). The BibTeX entries for these papers can be found [here (2021 paper)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504...65A/exportcitation) and [here (2025 paper)](https://ui.adsabs.harvard.edu/abs/2025arXiv250420299A/exportcitation).

License
-------

Copyright (c) 2021-2025 Abhijeet Anand.

**qsoabsfind** is a free software made available under the MIT License. For details, see the LICENSE file.

Thanks,  
Abhijeet Anand  
Lawrence Berkeley National Lab

