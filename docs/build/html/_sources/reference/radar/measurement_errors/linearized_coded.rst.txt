.. currentmodule:: sorts.radar.measurement_errors.linearized_coded

.. _linearized_coded:

=======================
Linearized coded errors
=======================

Description
-----------
The propagation and scattering of encoded radar waves causes errors to perturb range and doppler measurements. SORTS :ref:`linearized_coded` module performs a linear estimate of those errors using direct Monte-Carlo Simulations.

API Reference
-------------

1. Functions
~~~~~~~~~~~~
The :ref:`linearized_coded` module includes a set of functions to perform linear estimates of **range** and **doppler** errors.

.. autosummary::
	:toctree: auto/

	~simulate_echo
	~lin_error
	~precalculate_dr


2. Linear error estimation classes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :ref:`linearized_coded` module also provides the user with a set of standard error estimation classes which can be used to generate noisy date based on the estimated **range** and **doppler** errors. 

.. autosummary::
	:toctree: auto/

	~LinearizedCoded
	~LinearizedCodedIonospheric

Examples
--------
This simple example showcases the use of the :class:`LinearizedCoded` class to compute the perturbations 
in range measurements due to ``Signal-to-Noise Ratio`` fluctuations. 

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