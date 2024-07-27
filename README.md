# qsoabsfind

`qsoabsfind` is a Python module designed to detect absorbers with doublet properties in SDSS/DESI quasar spectra using a convolution method. This tool identifies potential absorbers, applies Gaussian fitting to reject false positives, and computes equivalent widths (EWs) of the lines. Currently the package works for MgII 2796,2803 and CIV 1548,1550 doublets

## Features

- Convolution-based detection of absorbers in QSO spectra.
- Gaussian fitting for accurate measurement of absorber properties.
- Parallel processing for efficient computation on large datasets.

## Installation

### Prerequisites

- Python 3.6 or higher
- `numpy`
- `scipy`
- `astropy`
- `fitsio`
- `numba`
- `pytest` (for running tests)

### Clone the Repository

First, clone the repository to your local machine:

```sh
git clone https://github.com/abhi0395/qsoabsfind.git
cd qsoabsfind
pip install .
python -m unittest discover -s tests

```

### Running example:
```sh
qsoabsfind --input-fits-file path_to_input_fits_file.fits \
           --n-qso 1-1000:10 \
           --absorber MgII \
           --constant-file path_to_constants.py \
           --output path_to_output_fits_file.fits \
           --headers KEY1=VALUE1 KEY2=VALUE2 \
           --n-tasks 16 \
           --ncpus 4
```

## Contribution
Contributions are welcome! Please submit a pull request or open an issue to discuss your ideas.

Thanks,
Abhijeet Anand
Lawrence Berkeley National Lab

In case any questions/suggestions, please feel free to write to abhijeetanand2011@gmail.com
