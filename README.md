<div align="center">
    <img src="https://raw.githubusercontent.com/abhi0395/qsoabsfind/HEAD/docs/images/logo.png" width="450" height="305"/>
</div>

<br>

<div align="center">

[![GitHub tag](https://img.shields.io/github/v/tag/abhi0395/qsoabsfind?sort=semver)](https://github.com/abhi0395/qsoabsfind/tags)
[![arXiv-2103.15842](http://img.shields.io/badge/arXiv-2103.15842-orange.svg?style=flat)](https://arxiv.org/abs/2103.15842)
[![arXiv-2504.20299](http://img.shields.io/badge/arXiv-2504.20299-orange.svg?style=flat)](https://arxiv.org/abs/2504.20299)
[![Tests](https://github.com/abhi0395/qsoabsfind/actions/workflows/tests.yml/badge.svg)](https://github.com/abhi0395/qsoabsfind/actions)
[![codecov](https://codecov.io/gh/abhi0395/qsoabsfind/branch/main/graph/badge.svg)](https://codecov.io/gh/abhi0395/qsoabsfind)
[![Documentation Status](https://readthedocs.org/projects/qsoabsfind/badge/?version=latest)](https://qsoabsfind.readthedocs.io/en/latest/?badge=latest)
[![license shields.io](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/abhi0395/qsoabsfind/blob/main/LICENSE)

</div>

## qsoabsfind: Quasar Absorber Finder

`qsoabsfind` is a Python module designed to detect absorbers with doublet properties in **SDSS** and **DESI** like low-resolution quasar spectra. It identifies potential absorption systems using a convolution-based, adaptive signal-to-noise approach, followed by Gaussian fitting and a series of rigorous checks to eliminate false positives.

The module also calculates rest-frame equivalent widths (EWs), FWHM and line centers using a double-Gaussian model. Optionally, it can calculate the total column densities of metal absorbers using the apparent optical depth method (AODM). The code offers flexibility to run with either default search parameters or user-provided custom search parameters.

### Supported Metal Doublets Systems

| Absorber | Line 1 (Å)  | Line 2 (Å) |
|----------|--------|------------|
| O VI (O⁵⁺)    | 1031.93    | 1037.62     |
| N V (N⁴⁺)    | 1238.82    | 1242.80     |
| Si IV (Si³⁺)   | 1393.76    | 1402.77     |
| C IV (C³⁺)   | 1548.20    | 1550.77     |
| Al III (Al²⁺)   | 1854.72     | 1862.79    |
| Fe II (Fe⁺)  | 2586.65     | 2600.17     |
| Mg II (Mg⁺)   | 2796.35    | 2803.52     |


Key Features
--------
- **Automated and Flexible Search Window**: The code can dynamically define the observed-frame wavelength search window for each absorber system. Detailed definitions are provided in the [Search Window Documentation](https://qsoabsfind.readthedocs.io/en/latest/searchwindows.html). Additionally, user can also provide the wavelength boundaries to search for metal systems through the search parameter constants file.
- **Flexible Search Parameters:** Supports both default settings and user-provided custom search parameters for metal absorber detection. Please use the constant file format as described in ``data/${survey}`` folder.
- **Adaptive S/N convolution**: Detects doublet absorbers in low-resolution quasar spectra using a convolution-based, adaptive signal-to-noise method.
- **Gaussian profile fitting**: Accurately models absorption lines to extract parameters like equivalent width, FWHM, and central wavelength.
- **Rigorous selection criteria**: Identifies the best absorber candidates based on physically motivated thresholds and doublet properties. Optionally uses chi2 statistics to get the confidence level of the selected candidates.
- **Instrumental resolution correction**: Corrects measured line widths for instrumental resolution to infer intrinsic properties.
- **Column Densities**: Optionally estimates total column densities of detected absorbers using the apparent optical depth method (AODM; [Savage & Sembach 1991](https://ui.adsabs.harvard.edu/abs/1991ApJ...379..245S/abstract)).
- **Parallel processing**: Supports fast and efficient computation across large datasets using Python's `multiprocessing` module.
- **Comprehensive Output**: Detailed catalogs with redshifts, equivalent widths, S/N ratios, and more.
- **Descriptive Verbose**: Optionally prints the steps in great detail for debugging.


Documentation
-------------

The full documentation is available at [https://qsoabsfind.readthedocs.io](https://qsoabsfind.readthedocs.io).

## Installation

### Prerequisites

- Python 3.6 or higher
- `numpy`
- `scipy`
- `astropy`
- `numba`
- `matplotlib`
- `pytest` (for running tests)

### 1. Clone the Repository
```bash

git clone https://github.com/abhi0395/qsoabsfind.git
cd qsoabsfind
```

### 2. Set Up Environment

#### Option 1: Using Conda (Recommended, python>=3.9)

```bash
conda create -n qsoabsfind python=3.9
conda activate qsoabsfind

# Install dependencies
conda install numpy scipy astropy numba matplotlib
conda install -c conda-forge pytest
```

#### Option 2: Using pip with virtual environment

```bash
python -m venv qsoabsfind-env
source qsoabsfind-env/bin/activate  

# Install dependencies
pip install numpy scipy astropy numba matplotlib pytest

### Install Package

pip install .

# For developers (editable mode installation):
pip install -e .
```

### 3. Run Unit tests

```bash
python -m unittest discover -s tests
```

### 4. Quick installation test
```bash
python -c "from qsoabsfind.parallel_convolution import parallel_convolution_method_absorber_finder_QSO_spectra; print('Installation successful!')"
```

Description
-----------

```sh
qsoabsfind --help
```

Important Instructions
-------------

- Before running the module, please read the [datamodel](https://github.com/abhi0395/qsoabsfind/blob/main/data/datamodel.rst). The instructions for the input and output files are provided there.
- I have also provided two example QSO spectra files:
  -  `data/sdss/qso_test_spectra.fits` : 100 continuum-normalized spectra from [SDSS DR16](https://www.sdss4.org/dr17/algorithms/qso_catalog/)
  -  `data/desi/qso_test_spectra.fits` : 100 continuum-normalized spectra from [DESI DR1](https://data.desi.lbl.gov/doc/releases/dr1/)
- These folders also have their own constants files. You can use these files to test example runs as described below.

Running as bash script:
----------------

**SDSS DR16 Spectra (without column densities)**
---------------------

```sh
qsoabsfind --input-fits-file data/sdss/qso_test_spectra.fits \
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

Citation & Acknowledgements
---------------------------

If you use this code in your analysis, please cite [Anand, Nelson & Kauffmann 2021](https://arxiv.org/abs/2103.15842) and [Anand et al. 2025](https://arxiv.org/abs/2504.20299). The BibTeX entries for these papers can be found [here (2021 paper)](https://ui.adsabs.harvard.edu/abs/2021MNRAS.504...65A/exportcitation) and [here (2025 paper)](https://ui.adsabs.harvard.edu/abs/2025arXiv250420299A/exportcitation).

Additionally, please also cite the associated [Zenodo record](https://zenodo.org/records/15685771). Additionally, consider starring the repository if you find it useful or use it in your work.

You can also copy the BibTeX entry directly from below.

```bibtex

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
  author       = {{Anand}, Abhijeet},
  title        = "{qsoabsfind: A Python Package for Detecting
                  Absorption Line Doublets in SDSS and DESI Quasar
                  Spectra}",
  month        = jun,
  year         = 2025,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.15685771},
  url          = {https://doi.org/10.5281/zenodo.15685771},
}

```
Contribution
------------

Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas. If you have any questions/suggestions, please feel free to write to **abhijeetanand2011@gmail.com** or, preferably, open a GitHub issue.

License
-------

Copyright (c) 2021-2025 Abhijeet Anand.

**qsoabsfind** is a free software made available under the MIT License. For details, see the LICENSE file.

Thanks,  
Abhijeet Anand  
IUCAA, Pune &
Lawrence Berkeley National Lab




