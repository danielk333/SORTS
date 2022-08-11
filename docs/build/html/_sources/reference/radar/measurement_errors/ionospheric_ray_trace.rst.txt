.. currentmodule:: sorts.radar.measurement_errors.ionospheric_ray_trace

.. _ionospheric_rt_errors:

==============================
Ionospheric Ray Tracing Errors
==============================

Description
-----------
At high altitudes, the propagation of radar signals is perturbed by the ionized plasma present in the Ionosphere. Those perturbations lead to two main sources of error :

 * 	**Electromagnetic wave refraction :** Refractive index gradients (caused by fluctuations in the plasma electron density) lead to the refraction of radar beams in the upper layers of the atmosphere. This refraction causes a local curvature of the beam which in turn modifies the relative angle between the *apparent* and *real positions* of the space object from the radar perspective.   
 *  **Propagation time delay :** The modification of the refractive index imply local modifications of the speed of light in the medium, leading to greater propagation time than predicted under the **free-space propagation** assumption.

To compute the ionospheric errors, it is necessary to perform a path integral (which is effectively summing all the local fluctuations in *velocity* and *curvature* of the beam) over the trajectory of the radar pulse in space. This technique is called Ray-Tracing.

API Reference
-------------
1. Functions
~~~~~~~~~~~~
The :ref:`ionospheric_rt_errors` module implements a set of functions to easily perform raytracing simulations using the Pyglow python library.

.. autosummary::
	:toctree: auto/

	~calculate_delay
	~ray_trace
	~ray_trace_error
	~ionospheric_error



2. Ionospheric Ray Tracing class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TODO

.. warn:: 
	Not fully implemented yet.

.. autosummary::
	:toctree: auto/

	~IonosphericRayTrace

Examples
--------
This example showcases the use of the :ref:`ionospheric_rt_errors` to compute the trajectory of the perturbed radar signal in space.

.. code-block::

	import numpy as np
	import matplotlib.pyplot as plt
	from astropy.time import Time
	from mpl_toolkits.mplot3d import Axes3D

	import sorts

	# compute raytracing results usign pyglow
	results = sorts.measurement_errors.ray_trace(
	        time = Time('2004-6-21 12:00'),
	        lat = 69.34023844,
	        lon = 20.313166,
	        frequency=233e6,
	        elevation=30.0,
	        azimuth=180.0,
	)

	# plot results
	fig=plt.figure(figsize=(14,8))
	plt.clf()
	plt.subplot(131)
	plt.title("Elevation=%1.0f"%(30.0))
	plt.plot(np.sqrt(
	    (
	        results['p0x']-results['px'])**2.0
	     + (results['p0y']-results['py'])**2.0
	     + (results['p0z']-results['pz'])**2.0
	     ),results['altitudes'],label="Total error")

	plt.plot(results['altitude_errors'],results['altitudes'],label="Altitude error")
	plt.ylim([0,1900])

	plt.grid()
	plt.legend()
	plt.xlabel("Position error (m)")
	plt.ylabel("Altitude km")

	plt.subplot(132)
	plt.plot(results['ray_bending']*1e6,results['altitudes'])

	plt.xlabel("Ray-bending ($\mu$deg/km)")
	plt.ylabel("Altitude km")
	plt.title("Total error=%1.2g (deg)"%(180.0*results['total_angle_error']/np.pi))
	plt.ylim([0,1900])
	plt.subplot(133)
	plt.plot(results['electron_density'],results['altitudes'])
	plt.xlabel("$N_{\mathrm{e}}$ ($\mathrm{m}^{-3}$)")
	plt.ylabel("Altitude km")
	plt.ylim([0,1900])

	plt.tight_layout()
	plt.show()