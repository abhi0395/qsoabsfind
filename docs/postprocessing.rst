Post-processing and Visualisation
==================================

Reading Output Catalogs
-----------------------

After running the absorber search, you can load the output FITS catalog using
the ``AbsorberData`` class:

.. code-block:: python

      from qsoabsfind.datamodel import AbsorberData

      catalog = AbsorberData('test_MgII.fits', autoload=True)

      print(catalog.catalog)        # absorber table (ABSORBER HDU)
      print(catalog.metadata)       # QSO metadata for detected absorbers (METADATA HDU)
      print(catalog.qso_info)       # all processed spectra with IS_QSO_AVAILABLE flag (QSO_INFO HDU)
      print(catalog.column_density) # column densities if --coldens was used, else None


Plotting a Random Absorber
--------------------------

Once you have loaded the spectra and the output catalog, you can visualise a
randomly selected absorber using :func:`qsoabsfind.utils.plot_absorber`:

.. code-block:: python

    import numpy as np
    from qsoabsfind.datamodel import QSOSpecRead, AbsorberData
    from qsoabsfind.utils import plot_absorber

    # Load the output absorber catalog
    catalog = AbsorberData('/path/to/your/absorber.fits', autoload=True)

    # Pick a random absorber from the catalog
    rng = np.random.default_rng()
    idx = rng.integers(len(catalog.catalog))
    row = catalog.catalog[idx]

    # Load the corresponding QSO spectrum
    spectra = QSOSpecRead('/path/to/your/spectra.fits',
                          index=int(row['INDEX_SPEC']),
                          autoload=True)

    # Plot the absorber (full spectrum + zoomed-in doublet view)
    plot_absorber(spectra, absorber='MgII', zabs=row,
                  title=f"MgII absorber at z={row['Z_ABS']:.4f}")

Pass ``show_error=True`` to overlay the error spectrum, or
``plot_filename='absorber.png'`` to save the figure to disk instead of
displaying it interactively.


Plotting All Absorbers Across Multiple Systems
----------------------------------------------

To visualise every detected system in a given spectrum, use
:func:`qsoabsfind.utils.plot_multiple_metal_systems`. It plots the full
spectrum with all systems annotated, followed by a zoomed panel for each
individual detection. It assumes that both catalogs are one to one mapped
to the same input spectra (e.g. both catalogs were generated from the same
input file) and uses the ``INDEX_SPEC`` column to match absorbers across
different systems:

.. code-block:: python

   from qsoabsfind.io import QSOSpecRead
   from qsoabsfind.utils import plot_multiple_metal_systems
   from astropy.io import fits
   from astropy.table import Table

   # Load spectrum
   spectra = QSOSpecRead('spectra.fits', qso_index=0)

   # Load catalogs for two absorbers
   with fits.open('output_MgII.fits') as hdul:
       mgii = Table(hdul['ABSORBER'].data)
   with fits.open('output_CIV.fits') as hdul:
       civ = Table(hdul['ABSORBER'].data)

   # Plot full spectrum + zoomed panels for every detected system
   plot_multiple_metal_systems(
       spectra,
       absorber_dict={'MgII': mgii, 'CIV': civ},
       zoom=True,
       plot_filename='absorbers.pdf',   # or None to display interactively
   )
