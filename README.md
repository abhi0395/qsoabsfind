qsoabsfind
============

**The Python module designed to detect absorbers with doublet properties in SDSS/DESI quasar**

<!-- [![github shields.io](https://img.shields.io/badge/GitHub-abhi0395%2Fqsoabsfind-blue.svg?style=flat)](https://github.com/abhi0395/qsoabsfind)-->
[![GitHub tag](https://img.shields.io/github/v/tag/abhi0395/qsoabsfind?sort=semver)](https://github.com/abhi0395/qsoabsfind/tags)
[![arXiv-2103.15842](http://img.shields.io/badge/arXiv-2103.15842-orange.svg?style=flat)](https://arxiv.org/abs/2103.15842)
[![arXiv-2504.20299](http://img.shields.io/badge/arXiv-2504.20299-orange.svg?style=flat)](https://arxiv.org/abs/2504.20299)
[![Tests](https://github.com/abhi0395/qsoabsfind/actions/workflows/tests.yml/badge.svg)](https://github.com/abhi0395/qsoabsfind/actions)
[![coverage](https://codecov.io/gh/abhi0395/qsoabsfind/branch/main/graph/badge.svg)](https://codecov.io/gh/abhi0395/qsoabsfind)
[![Documentation Status](https://readthedocs.org/projects/qsoabsfind/badge/?version=latest)](https://qsoabsfind.readthedocs.io/en/latest/?badge=latest)
[![license shields.io](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/abhi0395/qsoabsfind/blob/main/LICENSE)

`qsoabsfind` is a Python module designed to detect absorbers with doublet properties in **SDSS** and **DESI** like low-resolution quasar spectra. It identifies potential absorption systems using a convolution-based, adaptive signal-to-noise approach, followed by Gaussian fitting and a series of rigorous checks to eliminate false positives. The module also calculates rest-frame equivalent widths (EWs), FWHM and line centers using a double-Gaussian model. Optionally, it can calculate the total column densities of metal absorbers using the apparent optical depth method (AODM).

### Supported Metal Doublets Systems

| Absorber | Line 1 (Å) | oscillator strength 1     | Line 2 (Å) | oscillator strength 1     |
|----------|------------|--------|------------|--------|
| Mg II (Mg⁺)   | 2796.35    | 0.6123 | 2803.52    | 0.3054 |
| C IV (C³⁺)   | 1548.20    | 0.1900 | 1550.77    | 0.0952 |
| O VI (O⁵⁺)    | 1031.93    | 0.1329 | 1037.62    | 0.0661 |
| N V (N⁴⁺)    | 1238.82    | 0.1570 | 1242.80    | 0.0782 |
| Si IV (Si³⁺)   | 1393.76    | 0.5140 | 1402.77    | 0.2553 |
| Al III (Al²⁺)   | 1854.72    | 0.5390 | 1862.79    | 0.2680 |
| Fe II (Fe⁺)  | 2586.65    | 0.0691 | 2600.17    | 0.2394 |


Key Features
--------

- **Adaptive S/N convolution**: Detects doublet absorbers in low-resolution quasar spectra using a convolution-based, adaptive signal-to-noise method.
- **Rigorous selection criteria**: Identifies the best absorber candidates based on physically motivated thresholds and doublet properties.
- **Gaussian profile fitting**: Accurately models absorption lines to extract parameters like equivalent width, FWHM, and central wavelength.
- **Instrumental correction**: Corrects measured line widths for instrumental resolution to infer intrinsic properties.
- **Column Densities**: Optionally estimates total column densities of detected absorbers using the apparent optical depth method (AODM; [Savage & Sembach 1991](https://ui.adsabs.harvard.edu/abs/1991ApJ...379..245S/abstract)).
- **Automated Search Window**: The code dynamically defines the observed-frame wavelength search window for each absorber system. Detailed definitions are provided in the [Search Window Documentation](https://qsoabsfind.readthedocs.io/en/latest/searchwindows.html).
- **Parallel processing**: Supports efficient computation across large datasets using Python's `multiprocessing` module.


Documentation
-------------

The full documentation is available at [https://qsoabsfind.readthedocs.io](https://qsoabsfind.readthedocs.io).

Installation
------------

Prerequisites for installation
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

Important Instructions
-------------

- Before running the module, please read the [datamodel](https://github.com/abhi0395/qsoabsfind/blob/main/data/datamodel.rst). The instructions for the input and output files are provided there.
- I have also provided two example QSO spectra files:
  -  `data/sdss/qso_test_spectra.fits` : 500 continuum-normalized spectra from [SDSS DR16](https://www.sdss4.org/dr17/algorithms/qso_catalog/)
  -  `data/desi/qso_test_spectra.fits` : 500 continuum-normalized spectra from [DESI DR1](https://data.desi.lbl.gov/doc/releases/dr1/)
- You can use this file to test an example run as described below.

Running as bash script:
----------------

**SDSS DR16 Spectra (without column densities)**
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

**DESI DR1 Spectra (without column densities)**
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

**SDSS DR16 Spectra (with column densities)**
---------------------

```sh
qsoabsfind --input-fits-file data/sdss/qso_test_spectra.fits \
           --n-qso 500 \
           --absorber MgII \
           --output test_MgII.fits \
           --headers SURVEY=SDSS AUTHOR=YOUR_NAME \
           --ncpus 4 \
           --constant-file data/sdss/sdss_constants.py
           --coldens
           --dv 300
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

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas. If you have any questions/suggestions, please feel free to write to **abhijeetanand2011@gmail.com** or, preferably, open a GitHub issue.

Citation
--------

If you use this code in your analysis, please cite [Anand, Nelson & Kauffmann 2021](https://arxiv.org/abs/2103.15842) and [Anand et al. 2025](https://arxiv.org/abs/2504.20299). The BibTeX entries for these papers can be found [here (2021 paper)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504...65A/exportcitation) and [here (2025 paper)](https://ui.adsabs.harvard.edu/abs/2025arXiv250420299A/exportcitation).

If you use this **codebase**, please also cite the associated [Zenodo record](https://zenodo.org/records/15685771). Additionally, consider starring the repository if you find it useful or use it in your work.


License
-------

Copyright (c) 2021-2025 Abhijeet Anand.

**qsoabsfind** is a free software made available under the MIT License. For details, see the LICENSE file.

Thanks,
Abhijeet Anand
Lawrence Berkeley National Lab


