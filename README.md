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

- Convolution-based adaptive S/N approach for detecting absorbers in QSO spectra.
- Gaussian fitting for accurate measurement of absorber properties (such as EW, line widths, and centers).
- Parallel processing using multiprocessing for efficient computation on a large number of spectra.

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

Before running the program, please read the `data/datamodel.rst` file. The instructions for the input and output files are provided there. I have also provided an example QSO spectra FITS file, `data/qso_test.fits`, which contains 500 continuum-normalized SDSS QSO spectra. You can use this file to test an example run as described below.

Running example:
----------------

```sh
qsoabsfind --input-fits-file data/qso_test.fits \
           --n-qso 500 \
           --absorber MgII \
           --output test_MgII.fits \
           --headers SURVEY=SDSS AUTHOR=YOUR_NAME \
           --n-tasks 16 \
           --ncpus 4
```

Useful notes:
-------------

Parallel mode can be memory-intensive if the input FITS file is large in size. As the code accesses the FITS file to read QSO spectra when running in parallel, it can become a bottleneck for memory, and the code may fail. Currently, I suggest the following:

- **Divide your file into smaller chunks:** Split the FITS file into several smaller files, each containing approximately `N` spectra. Then run the code on these smaller files.

- **Use a rule of thumb for file size:** Ensure that the size of each individual file is no larger than `total_memory/ncpu` of your node or system. Based on this idea you can decide your `N`. I would suggest `N = 1000`.

- **Merge results at the end:** After processing, you can merge your results.

In order to decide the right size of the FITS file, consider the total available memory and the number of CPUs in your system.

Contribution
------------

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas. If you have any questions/suggestions, please feel free to write to abhijeetanand2011@gmail.com or, preferably, open a GitHub issue.

Citation
--------

Please cite [Anand, Nelson & Kauffmann 2021](https://arxiv.org/abs/2103.15842) if you find this code useful in your research. The BibTeX entry for the paper can be found [here](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504...65A/exportcitation).


License
-------

Copyright (c) 2021-2025 Abhijeet Anand.  

**qsoabsfind** is a free software made available under the MIT License. For details, see the LICENSE file.


Thanks,  
Abhijeet Anand  
Lawrence Berkeley National Lab  
