Absorber Search Windows
=======================

For each absorber, user can define an observed-frame wavelength search window based on the quasar redshift and proximity to major emission lines. This ensures that:

- We avoid regions near the quasar's own emission lines where intrinsic absorption may contaminate our detection.
- We stay within the observed wavelength coverage.
- We apply wavelength offset near the edges to make sure both lines are still inside the wavelength window and edges are not very noisy.

Parameter Definitions
---------------------

- :math:`\lambda_{\mathrm{min}},\, \lambda_{\mathrm{max}}`: Full observed-frame wavelength coverage of the spectrum.
- :math:`z_{\mathrm{QSO}}`: Emission redshift of the background quasar.
- :math:`\Delta z` is derived from the velocity offset parameter (``search_parameters["dv"]``, from user's constant file) to avoid regions close to emission lines.
- :math:`\Delta\lambda_{\mathrm{edge}}` is a offset from edges to avoid noisy regions during convolution or feature detection near the wavelength edges.


Search Window Definitions
-------------------------

For each absorber, the search window in observed-frame wavelength is defined using quasar intrinsic emission-line and a velocity-based offset. The purpose is to restrict the search to physically motivated regions around the quasar where the absorber is likely to appear, while avoiding contamination from unrelated features.

The offset is determined via a velocity range (e.g., +/-5000 km/s), converted to redshift as:

The general form for the observed-frame wavelength of a line is:

.. math::

   \lambda_{\mathrm{obs}} = \lambda_{\mathrm{rest}} \times (1 + z)

To convert a velocity offset :math:`\Delta v` (in km/s) into a redshift offset:

.. math::

   \Delta z = \frac{|\Delta v|}{c} \times (1 + z_{\mathrm{QSO}})

where :math:`c` is the speed of light.

User-defined wavelength boundaries (in quasar rest-frame):
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The code allows users to provide the wavelength boundaries (in the quasar rest-frame) to define the absorber search window. Details are provided below.

- **Emission lines**:
    - Blue side: ``search_parameters["start_rest_wave"]``
    - Red side: ``search_parameters["end_rest_wave"]``

- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\rm start}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\rm end}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}


Default search window logic used for each metal doublet:
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Mg II (2796, 2803 Ang)
~~~~~~~~~~~~~~~~~~~~~~

- **Emission lines**: blue side - C IV (1549.5 Ang), red side - Mg II (2799.1 Ang)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{CIV}}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{MgII}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

C IV (1548, 1550 Ang)
~~~~~~~~~~~~~~~~~~~~~

- **Emission lines**: blue side - Outside Si II forest (>1310 Ang), red side - C IV (1549.5 Ang)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}}, 1310 \times (1 + z_{\mathrm{QSO}}+ \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{CIV}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

O VI (1032, 1038 Ang)
~~~~~~~~~~~~~~~~~~~~~

- **Emission lines**: blue side -  500 Ang (arbitrary low), red side - O VI (1033.82 Ang)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, 500\times(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{OVI}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

N V (1238, 1242 Ang)
~~~~~~~~~~~~~~~~~~~~

- **Emission lines**: blue side - Ly-beta (1025.72 Ang), red side - N V (1240.8 Ang)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{Ly}\beta}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{NV}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

Si IV (1394, 1403 Ang)
~~~~~~~~~~~~~~~~~~~~~~

- **Emission lines**: blue side - Ly-alpha (1215.67 Ang), red side - Si IV (1399.8 Ang)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{Ly}\alpha}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{SiIV}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

Al III (1854, 1862 Ang)
~~~~~~~~~~~~~~~~~~~~~~~

- **Emission lines**: blue side - C IV (1549.5 Ang), red side - Al III (1857.4 Ang)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{CIV}}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{AlIII}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

Fe II (2586, 2600 Ang)
~~~~~~~~~~~~~~~~~~~~~~

- **Emission lines**: blue side - C IV (1549.5 Ang), red side - Mg II (2799.1 Ang)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{CIV}}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{MgII}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

Ca II (3934, 3969 Ang)
~~~~~~~~~~~~~~~~~~~~~~

- **Emission lines**: blue side - Ly-alpha (1215.67 Ang), red side - 9800 Ang (arbitrary high)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{Ly}\alpha}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, 9800 \times (1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}


Na I (5891, 5897 Ang)
~~~~~~~~~~~~~~~~~~~~~

- **Emission lines**: blue side - Ly-alpha (1215.67 Ang), red side - 9800 Ang (arbitrary high)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{Ly}\alpha}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, 9800 \times (1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

