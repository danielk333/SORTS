.. currentmodule:: sorts.radar.signals

.. _signals:

=============================
Signals (sorts.radar.signals)
=============================

Description
-----------
SORTS :ref:`signals` module implements a set of signal simulation functions which allow to calculate radar-related properties of space objects (such as their `Radar Cross-Section (RCS)`, `Signal-to-Noise Ratio (SNR)`...). 

.. _signals-theory:

Theoretical background
----------------------

.. _signals-theory-radar_eq:

The radar equation
~~~~~~~~~~~~~~~~~~
The radar equation describes the fundamental relationship between the characteristics of the radar, the target, the propagation medium and the received signal. Under the assumption of free-space propagation, it is possible to express the power of the received radar signal :math:`P_{s, rx}` at the antenna terminals as a function of the peak transmission power :math:`P_{tx}`, the gain of the receiving :math:`G_{rx}` and transmitting :math:`G_{tx}` antennas, the :ref:`signals-theory-rcs` of the target :math:`\sigma`, wavelength of the radar wave :math:`\lambda` and the range of the target :math:`R` as:

.. math::
						P_{s, rx} = \frac{P_{tx} G_{tx} G_{rx} \sigma_{t} \lambda^2}{(4 \pi)^3 R^4}


This expression relies deeply on the correct estimation of the :ref:`signals-theory-rcs`


.. _signals-theory-rcs:

Radar Cross-Section (RCS)
~~~~~~~~~~~~~~~~~~~~~~~~~
The implementation of this module relies on the  Radar Cross-Section of an object can be estimated using the radar cross section of a dielectric sphere. There are multiple scattering regimes depending on the radar wavelength (Optical, Mie, Rayleigh). For simplification, only the optical and Rayleigh scattering regimes are considered in the current implementation. Using those assumption, it is possible to write: 

.. math::
					\sigma = \left\{
					    \begin{array}{ll}
					        \hat{\sigma} \frac{9 \pi^5}{4 \lambda^4} D^6   	& \mbox{if } D < \frac{\lambda}{\pi \sqrt{3}} \\
					        \hat{\sigma} \frac{\pi}{4} D^2  				& \mbox{if } D \geq \frac{\lambda}{\pi \sqrt{3}}
					    \end{array}
					\right.

With :math:`\hat{\sigma} \approx \| (\epsilon_r - 1)/(\epsilon_r + 2) \|^2` the radar albedo of the target, :math:`\lambda` the radar wavelength and :math:`D` the diameter of the equivalent sphere modelling the object.

.. figure:: ../../../figures/rcs_diameter.png
   :alt: Normalized RCS as a function of the normalized object diameter. 

   Normalized RCS as a function of the normalized object diameter. The transition between the two scattering regimes can be observed at :math:`D/\lambda = 1/ \pi \sqrt{3}`.

.. note::
	This model does not describe the variations in **SNR** caused by the geometry and spacial orientation of uneaven rotating objects. Some models [1]_ were developped to account for the shape of the space debris but those are not currently implemented in SORTS.

.. _signals-theory-snr:

Signal-to-Noise Ratio (SNR)
~~~~~~~~~~~~~~~~~~~~~~~~~~~
The signal :ref:`signals-theory-snr` corresponds to the ratio of the *measured signal power* to the *background noise power*:

.. math::			SNR = \frac{P_s}{P_n}

It is common to characterize the statistical background noise :math:`P_n` with an equivalent **Noise temperature** :math:`T_n`, such as:

.. math::			P_n = \beta k_B T_n

with :math:`\beta` the receiver bandwidth and :math:`k_B` the Boltzman constant.


.. _signals-functions:

Signals API
-----------
The :ref:`signals` module implements the following functions:

.. autosummary::
	:toctree: auto/

	~doppler_spread_hard_target_snr
	~hard_target_diameter
	~hard_target_rcs
	~hard_target_snr
	~hard_target_snr_scaling
	~incoherent_snr

.. _signals-examples:

Examples
--------
A simple example using the features of the :ref:`signals` module consits in calculating the **SNR** (coherent/incoherent) for a specific space object/radar arrangement. This SNR value can then be compated to a threshold value to determine if the space object would have been detected with the current system configuration.

.. code-block:: Python

	import numpy as np
	import sorts

	# intitializes the radar
	radar = sorts.radars.eiscat3d

	# point stations towards local vertical
	k0 = np.array([0,0,1])
	radar.tx[0].beam.point(k0)
	radar.rx[0].beam.point(k0)

	# compute incoherent and coherent SNR
	snr_coh, snr_incoh = sorts.signals.doppler_spread_hard_target_snr(
	    3600.0, 
	    gain_tx = radar.tx[0].beam.gain(k0),
	    gain_rx = radar.rx[0].beam.gain(k0),
	    wavelength = radar.tx[0].wavelength,
	    power_tx = radar.tx[0].power,
	    range_tx_m = 300000e3, 
	    range_rx_m = 300000e3,
	    duty_cycle=0.25,
	    bandwidth=10,
	    rx_noise_temp=150.0,
	    diameter=150.0,
	    spin_period=500.0,
	    radar_albedo=0.1,
	)

.. _signals-see-also:

See also
--------
SORTS also implements equivalent :ref:`signals-functions` C functions (see :ref:`c_lib-signals`) which can be used for **high-performance computations C implementations** without relying on **callbacks** to the python interpreter. They are for example used in the C functions of the :ref:`measurements` module.

..  warning::
	The *signals* functions implemented in C and in Python are independant. The python functions, implemented using numpy-vectorized computations, provide a more flexible tool for tests/simulations (since any parameter can be an array). On the other hand, the C functions have been optimized with respect to the :ref:`measurements` module requirements.

	Therefore, any modification made to one implementation must also be done on the other. 

.. _signals-references:

References
----------

 .. [1] Shape of space debris as estimated fron radar cross section varaitions, T. Sato et al., 1994, Journal of Spacecraft and Rockets, `http://www-lab26.kuee.kyoto-u.ac.jp/~tsato/publ-pdf/jsr94.pdf`_