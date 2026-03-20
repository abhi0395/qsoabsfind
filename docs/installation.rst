Installation
============

Prerequisites
-------------

- Python 3.10 or higher
- ``numpy``
- ``scipy``
- ``astropy``
- ``numba``
- ``matplotlib``
- ``tqdm`` (for progress bars)
- ``pyyaml`` (for configuration file handling)

Clone the Repository
--------------------

First, clone the repository to your local machine:

.. code-block:: bash

    git clone https://github.com/abhi0395/qsoabsfind.git
    cd qsoabsfind

Set Up Environment
------------------

**Option 1: Using Conda (Recommended, python>=3.9)**

Create and activate a conda environment:

.. code-block:: bash

    # Create environment with Python 3.9
    conda create -n qsoabsfind python=3.9
    conda activate qsoabsfind

    # Install dependencies
    conda install numpy scipy astropy numba matplotlib
    conda install -c conda-forge pytest

**Option 2: Using pip with virtual environment**

.. code-block:: bash

    # Create virtual environment
    python -m venv qsoabsfind-env
    source qsoabsfind-env/bin/activate  # Linux/Mac

    # Install dependencies
    pip install numpy scipy astropy numba matplotlib pytest

Install Package
---------------

Install qsoabsfind (tagged version, stable and reproducible):

.. code-block:: bash

    pip install --upgrade "git+https://github.com/abhi0395/qsoabsfind.git@vX.Y.Z"

Replace vX.Y.Z with the desired Git tag (for example v2.0.1).

For developers (editable mode installation):

.. code-block:: bash

    pip install -e .

Run Unit tests
--------------

Verify the installation by running tests:

.. code-block:: bash

    python -m unittest discover -s tests


Quick installation test
-----------------------

Test the installation:

.. code-block:: python

    python -c "import qsoabsfind; print(qsoabsfind.__version__)"
    python -c "from qsoabsfind.parallel_convolution import parallel_convolution_search; print('Installation successful!')"


Description
-----------

To see available options and usage:

.. code-block:: bash

    qsoabsfind --help