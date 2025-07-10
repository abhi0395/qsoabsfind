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

Description
---------

.. code-block:: bash

    qsoabsfind --help


Setting Environment Variable
------------------------

Before using the module, please set an environment variable `QSO_CONSTANTS_FILE` in your `bashrc` or `zshrc` file, and point it to the `qsoabsfind.constants` file. Since the code dynamically loads constants from a new file, it is important to define this environment variable.