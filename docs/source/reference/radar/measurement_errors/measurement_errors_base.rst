.. currentmodule:: sorts.radar.measurement_errors

.. _measurement_errors:

===================================================
Measurement Errors (sorts.radar.measurement_errors)
===================================================

Description
-----------
By default, SORTS simulations assume free-space propagation of the signals and ignore sources of noise outside of the receiver noise (see :attr:`noise temperature<sorts.radar.system.station.RX.noise_temperature>`). To reduce the impact of those strong assumptions, SORTS :ref:`measurement_errors` module includes a set of functions and classes which defines a standard interface to add random measurement errors to data and propagate uncertainties.

API Reference
-------------

1. The Errors class
~~~~~~~~~~~~~~~~~~~
The :class:`errors.Errors` class defines a standardized interface to add random errors to data. It is the base class from which all the other errorr-related classes inherit. 

.. autosummary::
	:toctree: auto/ 

	~errors.Errors

2. Signal propagation errors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SORTS :ref:`signals` module assumes free-space propagation of electromagnetic waves. This assumption, while greatly simplifying computations for simple proof-of-concept applications, is not sufficient to simulate real-world radar performances and observations. Therefore, SORTS also inculdes a set of **signal propagation error simulations** modules :

.. toctree::
	:maxdepth: 1

	linearized_coded
	ionospheric_ray_trace

Example
-------
This simple example showcases the use of the :class:`LinearizedCoded<linearized_coded.LinearizedCoded>` class to compute the perturbations in range measurements due to ``Signal-to-Noise Ratio`` fluctuations. 

.. code-block:: Python

	import sorts
	import numpy as np
	import matplotlib.pyplot as plt

	radar = sorts.radars.eiscat3d

	# initialization of the linearized errors for coded signals
	err = sorts.measurement_errors.LinearizedCoded(radar.tx[0], seed=123)

	# number of ranges
	num = 1000
	# number of range bins for posterior distribution estimate
	n_bins = 50

	# generate 100 range values and 
	ranges = np.linspace(300e3, 350e3, num=num)[::-1]

	# generate random SNR values
	snrs = np.random.randn(num)*15.0 + 20**1.0
	snrs[snrs<0.1] = 0.1

	# compute perturbated range estimates due to SNR fluctuations
	perturbed_ranges = err.range(ranges, snrs)

	# plot results
	fig, axes = plt.subplots(3, 1)
	axes[0].plot(np.arange(0, num), ranges, "--k") # initial range values
	axes[0].plot(np.arange(0, num), perturbed_ranges, "-r") # perturbed range values
	axes[0].set_xlabel("$N$ [$-$]")
	axes[0].set_ylabel("$r$ [$m$]")

	axes[1].hist(10*np.log10(snrs), n_bins, color="blue") # snr distribution
	axes[1].set_ylabel("$N$ [$-$]")
	axes[1].set_xlabel("$SNR$ [$dB$]")

	axes[2].hist(ranges - perturbed_ranges, n_bins, color="blue") # range error distribution 
	axes[2].set_ylabel("$N$ [$-$]")
	axes[2].set_xlabel("$r-r_{est}$ [$m$]")
	plt.show()

.. rubric:: results

.. figure:: ../../../figures/errors_example_ranges.png
