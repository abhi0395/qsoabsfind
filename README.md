qsoabsfind
============

**The Python module designed to detect absorbers with doublet properties in SDSS/DESI quasar**

[![github shields.io](https://img.shields.io/badge/GitHub-abhi0395%2Fqsoabsfind-blue.svg?style=flat)](https://github.com/abhi0395/qsoabsfind)
[![Tests](https://github.com/abhi0395/qsoabsfind/actions/workflows/tests.yml/badge.svg)](https://github.com/abhi0395/qsoabsfind/actions)
[![license shields.io](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/abhi0395/qsoabsfind/blob/main/LICENSE)
[![arXiv-2103.15842](http://img.shields.io/badge/arXiv-2103.15842-orange.svg?style=flat)](https://arxiv.org/abs/2103.15842)

`qsoabsfind` is a Python module designed to detect absorbers with doublet properties in SDSS/DESI quasar spectra. This tool identifies potential absorbers using a convolution-based adaptive S/N approach, applies Gaussian fitting and extensive checks to reject false positives, and computes equivalent widths (EWs) of the lines using a simple double Gaussian.

Currently, the package only works for **MgII 2796,2803** and **CIV 1548,1550** doublets.

Features
--------

- Convolution-based adaptive S/N approach for detecting absorbers in QSO spectra.
- Gaussian fitting for accurate measurement of absorber properties (such as EW, line widths, and centers).
- Parallel processing for efficient computation on a large number of spectra.

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

Before you run, please read the `datamodel.rst` file. The instructions about the input and output file are provided there.

Running example:
----------------

```sh
qsoabsfind --input-fits-file path_to_input_fits_file.fits \
           --n-qso 1-1000:10 \
           --absorber MgII \
           --output path_to_output_fits_file.fits \
           --headers KEY1=VALUE1 KEY2=VALUE2 \
           --n-tasks 16 \
           --ncpus 4
```

Contribution
------------

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

Citation
--------

Please cite [Anand, Nelson & Kauffmann 2021](https://arxiv.org/abs/2103.15842) if you find this code useful in your research. The BibTeX entry for the paper is:

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


License
-------

Copyright (c) 2021-2025 Abhijeet Anand.  

**qsoabsfind** is a free software made available under the MIT License. For details, see the LICENSE file.


Thanks,  
Abhijeet Anand  
Lawrence Berkeley National Lab  

If you have any questions/suggestions, please feel free to write to abhijeetanand2011@gmail.com
