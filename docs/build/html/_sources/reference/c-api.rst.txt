.. _c_lib

=========
C library
=========

Description
-----------
SORTS implements a set of optimized C functions to perform the computationally expensive computations required by SORTS. Beware that in contrast to the modules defined in ``Python`` within the toolbox, those functions are not designed for easy modification, but rather for high computational efficiency. The functions from the C-Library can be called using the ``sorts.clibsorts`` module.

.. _c_lib-signals:

Signal simulation
-----------------
The **Signal** simulation C library is based on the :mod:`signals` module implemented in python (see documentation associated for more information about the underlying theory).

1. Constants 
````````````
The signals implementation uses two physical constants : 


.. doxygendefine:: BOLTZMAN_CONSTANT


.. doxygendefine:: C_VACUUM

2. Functions 
````````````
The functions implemented within the signals C library are the following :


.. doxygenfunction:: doppler_spread_hard_target_snr_vectorized
 

.. doxygenfunction:: doppler_spread_hard_target_snr


.. doxygenfunction:: hard_target_diameter


.. doxygenfunction:: hard_target_diameter_vectorized


.. doxygenfunction:: hard_target_snr


.. doxygenfunction:: hard_target_snr_vectorized


.. doxygenfunction:: incoherent_snr


Radar
-----








