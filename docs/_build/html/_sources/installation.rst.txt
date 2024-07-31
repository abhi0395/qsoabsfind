Installation
============

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

.. code-block:: bash

    git clone https://github.com/abhi0395/qsoabsfind.git
    cd qsoabsfind
    pip install .
    python -m unittest discover -s tests

Running example:
----------------

Before running, please read :doc:`File formats <fileformat>`.

.. code-block:: bash

    qsoabsfind --input-fits-file path_to_input_fits_file.fits \
               --n-qso 1-1000:10 \
               --absorber MgII \
               --output path_to_output_fits_file.fits \
               --headers KEY1=VALUE1 KEY2=VALUE2 \
               --n-tasks 16 \
               --ncpus 4
