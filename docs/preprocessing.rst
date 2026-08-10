Pre-filtering Searchable QSOs
==============================

Before running the full absorber search you can quickly flag which spectra
have a usable wavelength window for the absorber of interest.
``qsoabsfind.absorberutils.find_searchable_qsos`` performs this check
in parallel over the whole file and returns a two-column
``astropy.table.Table`` (``QSO_INDEX``, ``IS_GOOD``) that can be used to
build a parent sample:

.. code-block:: python

    from qsoabsfind.absorberutils import find_searchable_qsos

    parent = find_searchable_qsos(
        fits_file='spectra.fits',
        absorber='MgII',
        constant_file='my_constants.py',
        ncpus=8,        # parallel workers
        n_qso=None,     # None = all spectra; or '1-5000', '500', '1-5000:2'
        verbose=False,
    )

    # keep only searchable QSOs
    good = parent[parent['IS_GOOD']]
    print(f"{len(good)} / {len(parent)} QSOs have a searchable MgII window")

The function applies the same overridable-constants logic as the
main pipeline, so the filtering is consistent with the absorber search.
