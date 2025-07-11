Search Windows
==============

For each absorber, we define an observed-frame wavelength search window based on the quasar redshift and proximity to major emission lines. This ensures that:

- We avoid regions near the quasar's own emission lines where intrinsic absorption may contaminate our detection.
- We stay within the observed wavelength coverage.
- We apply wavelength offset near the edges to make sure both lines are still inside the wavelength window and edges are not very noisy.

Parameter Definitions
-----------

- :math:`\lambda_{\mathrm{min}},\, \lambda_{\mathrm{max}}`: Full observed-frame wavelength coverage of the spectrum.
- :math:`z_{\mathrm{QSO}}`: Emission redshift of the background quasar.
- :math:`\Delta z` is derived from the velocity offset parameter (`lines["dv"]`, from `constants.py`) to avoid regions close to emission lines.
- :math:`\Delta\lambda_{\mathrm{edge}}` is a offset from edges to avoid noisy regions during convolution or feature detection near the wavelength edges.


Search Window Definitions
-----------

For each absorber, the search window in observed-frame wavelength is defined using quasar intrinsic emission-line and a velocity-based offset. The purpose is to restrict the search to physically motivated regions around the quasar where the absorber is likely to appear, while avoiding contamination from unrelated features.

The offset is determined via a velocity range (e.g., ±5000 km/s), converted to redshift as:

The general form for the observed-frame wavelength of a line is:

.. math::

   \lambda_{\mathrm{obs}} = \lambda_{\mathrm{rest}} \times (1 + z)

To convert a velocity offset :math:`\Delta v` (in km/s) into a redshift offset:

.. math::

   \Delta z = \frac{|\Delta v|}{c} \times (1 + z_{\mathrm{QSO}})

where :math:`c` is the speed of light.

Below we summarize the logic used for each metal doublet:

Mg II (2796, 2803 Å)
--------------------

- **Emission lines**: blue side - C IV (1549.5 Å), red side - Mg II (2799.1 Å)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{CIV}}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{MgII}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

C IV (1548, 1550 Å)
--------------------

- **Emission lines**: blue side - Outside Si II forest (>1310 Å), red side - C IV (1549.5 Å)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}}, 1310 \times (1 + z_{\mathrm{QSO}}+ \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{CIV}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

O VI (1032, 1038 Å)
--------------------

- **Emission lines**: blue side - O VI (1033.82 Å), red side - Lyα (1215.67 Å)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{OVI}}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{Ly}\alpha}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

N V (1238, 1242 Å)
--------------------

- **Emission lines**: blue side - Lyβ (1025.72 Å), red side - N V (1240.8 Å)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{Ly}\beta}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{NV}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

Si IV (1394, 1403 Å)
---------------------

- **Emission lines**: blue side - Lyα (1215.67 Å), red side - Si IV (1399.8 Å)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{Ly}\alpha}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{SiIV}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

Al III (1854, 1862 Å)
----------------------

- **Emission lines**: blue side - C IV (1549.5 Å), red side - Al III (1857.4 Å)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{CIV}}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{AlIII}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

Fe II (2586, 2600 Å)
---------------------

- **Emission lines**: blue side - C IV (1549.5 Å), red side - Mg II (2799.1 Å)
- **Search window**:

.. math::

    \lambda_{\mathrm{start}} = \max\{\lambda_{\mathrm{min}},\, \lambda_{\mathrm{CIV}}(1 + z_{\mathrm{QSO}} + \Delta z)\} + \Delta\lambda_{\mathrm{edge}}

.. math::

    \lambda_{\mathrm{end}} = \min\{\lambda_{\mathrm{max}},\, \lambda_{\mathrm{MgII}}(1 + z_{\mathrm{QSO}} - \Delta z)\} - \Delta\lambda_{\mathrm{edge}}

