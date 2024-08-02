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

Before running, please read :doc:`File formats <fileformat>`. I have provided an example QSO spectra FITS file, `data/qso_test.fits`, which contains 500 continuum-normalized SDSS QSO spectra. You can use this file to test an example run as described below.

.. code-block:: bash

    qsoabsfind --input-fits-file data/qso_test.fits \
               --n-qso 500 \
               --absorber MgII \
               --output test_MgII.fits \
               --headers SURVEY=SDSS AUTHOR=YOUR_NAME \
               --n-tasks 16 \
               --ncpus 4

Useful notes:
-------------

Parallel mode can be memory-intensive if the input FITS file is large in size. As the code accesses the FITS file to read QSO spectra when running in parallel, it can become a bottleneck for memory, and the code may fail. Currently, I suggest the following:

   - **Divide your file into smaller chunks:** Split the FITS file into several smaller files, each containing approximately `N` spectra. Then run the code on these smaller files.

   - **Use a rule of thumb for file size:** Ensure that the size of each individual file is no larger than `total_memory/ncpu` of your node or system. Based on this idea you can decide your `N`. I would suggest `N = 1000`.

   - **Merge results at the end:** After processing, you can merge your results.

In order to decide the right size of the FITS file, consider the total available memory and the number of CPUs in your system.
