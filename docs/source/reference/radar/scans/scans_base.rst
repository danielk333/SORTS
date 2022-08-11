.. currentmodule:: sorts.radar.scans

.. _scans:

=========================
Scans (sorts.radar.scans)
=========================

Description
-----------
SORTS :ref:`scans` encapsulate timed sequences of radar pointing directions. 


API Reference
-------------

1. Base Scan class
~~~~~~~~~~~~~~~~~~
The :class:`scan.Scan` defines the *attributes* and *methods* common to all radar scans. As such, all :class:`scan.Scan` instances must inherit from this class to ensure compatibility with other SORTS subsystems (radar, controllers, ...).

.. autosummary::
	:toctree: auto/

	~scan.Scan 

2. Predifined radar scans
~~~~~~~~~~~~~~~~~~~~~~~~~
SORTS includes the following set of predifined radar scanning schemes:

.. autosummary::
	:toctree: auto/

	~bp.Beampark
	~fence.Fence 
	~plane.Plane 
	~random_fence.RandomFence 
	~random_uniform.RandomUniform 
	~uniform.Uniform 

Examples
--------
This short example showcases the instantiation and plotting of 4 predifined :class:`scan.Scan` instances.

.. code-block:: Python

	import sorts
	import matplotlib.pyplot as plt

	# define the radar sytem
	radar = sorts.radars.eiscat3d

	# Scan parameters
	N = 100
	t_end = 20
	dwell = t_end/N

	# create 4 new scans of different type
	uniform 	= sorts.scans.Uniform(min_elevation=30.0, dwell=dwell, sph_num=N)
	rand_uniform 	= sorts.scans.RandomUniform(min_elevation=30.0, dwell=dwell, cycle_num=N)
	plane 		= sorts.scans.Plane(min_elevation=30.0, altitude=2000e3, x_size=1000e3, y_size=1000e3, x_num=int(N**0.5), y_num=int(N**0.5), dwell=dwell, x_offset=0.0, y_offset=0.0)
	bp 		= sorts.scans.Beampark(azimuth=180.0, elevation=75.0, dwell=dwell)

	# create figure
	fig = plt.figure()
	ax1 = fig.add_subplot(221, projection='3d')
	ax2 = fig.add_subplot(222, projection='3d')
	ax3 = fig.add_subplot(223, projection='3d')
	ax4 = fig.add_subplot(224, projection='3d')

	# plot generated scans
	sorts.plotting.plot_scanning_sequence(uniform, station=radar.tx[0], earth=True, ax=ax1, plot_local_normal=True, max_range=1000e3)
	sorts.plotting.plot_scanning_sequence(rand_uniform, station=radar.tx[0], earth=True, ax=ax2, plot_local_normal=True, max_range=1000e3)
	sorts.plotting.plot_scanning_sequence(plane, station=radar.tx[0], earth=True, ax=ax3, plot_local_normal=True, max_range=1000e3)
	sorts.plotting.plot_scanning_sequence(bp, station=radar.tx[0], earth=True, ax=ax4, plot_local_normal=True, max_range=1000e3)

	ax1.set_title("Uniform scan")
	ax2.set_title("RandomUniform scan")
	ax3.set_title("Plane scan")
	ax4.set_title("Beampark scan")
	plt.show()

.. rubric:: Ouput : 

.. figure:: ../../../figures/scans_example.png