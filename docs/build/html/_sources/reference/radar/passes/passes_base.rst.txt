.. currentmodule:: sorts.radar.passes

.. _passes:

=================================
Radar passes (sorts.radar.passes)
=================================

Description
-----------
It is usual that the majority of propagated :class:`Space Objects<sorts.target.space_object.SpaceObject>` states are not visible by the radar stations. **Radar passes** significantly reduce the number of states over which we need to simulate radar measurements by only storing states within the *field of view* of one or multiple radar stations.


.. figure:: ../../../figures/radar_pass.png
   :alt: Normalized RCS as a function of the normalized object diameter. 
   :scale: 75%

   Example of a **radar pass** (in red) over the EISCAT_3D radar system. Only the states within the radar system are stored, the states outside of the FOV (in blue) are ignored.


After having propagated the states of a :class:`Space Object<sorts.target.space_object.SpaceObject>` in time, the :ref:`passes` module can be used to filter only the states visible by specific stations, therefore creating an `ensemble` of subsets of states. After all the **passes** are created, it is then possible to run computations over those subsets much faster than it would have taken over the full set of states. 


The pass class
--------------
The :class:`Pass` class defines the *attributes* and *methods* of a **radar pass**.

.. autosummary::
	:toctree: auto/

	~Pass 


Additional functions
--------------------
The :ref:`passes` module also includes a set of *functions* to facilitate the creation and handling of **passes**.

.. autosummary::
	:toctree: auto/

	~equidistant_sampling
	~find_passes
	~find_simultaneous_passes
	~group_passes	 


Examples
--------
In this example, we want to showcase the advantages of using **radar passes** when dealing with a large number of space object states. 

First we define the state propagator. In our case, we will use the Kepler propagator:

>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> import sorts
>>> Prop_cls = sorts.propagator.Kepler
>>> Prop_opts = dict(
...     settings = dict(
...         out_frame='ITRS',
...         in_frame='TEME',
...     ),
... )

We then define the radar system. For this example, we will use the predifined :ref:`radar_eiscat3d` radar system:

>>> radar = sorts.radars.eiscat3d

Then, we create a new space object:

>>> space_object = sorts.SpaceObject(
...         Prop_cls,
...         propagator_options = Prop_opts,
...         a = 7000e3, 
...         e = 0.0,
...         i = 78,
...         raan = 86,
...         aop = 0, 
...         mu0 = 50,
...         epoch = 53005.0,
...         parameters = dict(
...             d = 0.1,
...         ),
...     )
>>> print(space_object)
Space object 1: <Time object: scale='utc' format='mjd' value=53005.0>:
a    : 7.0000e+06   x : -7.9830e+05
e    : 0.0000e+00   y : 4.5663e+06
i    : 7.8000e+01   z : 5.2451e+06
omega: 0.0000e+00   vx: -1.4093e+03
Omega: 8.6000e+01   vy: -5.6962e+03
anom : 5.0000e+01   vz: 4.7445e+03
Parameters: C_D=2.3, m=1.0, C_R=1.0, d=0.1

By using the function ``passes.equidistant_sampling()`` we can create a set of equidistant points to propagate the space object's states over a time period of *10 days*:

>>> t_states = sorts.equidistant_sampling(
...     orbit=space_object.state, 
...     start_t=0, 
...     end_t=3600.0*10, 
...     max_dpos=10e3)
>>> object_states = space_object.get_state(t_states)
>>> object_states.shape
(6, 27166)

To get all the passes of the space object over the :ref:`radar_eiscat3d` radar system, we call:

>>> radar_passes = radar.find_passes(t_states, object_states, cache_data=False) 
>>> radar_passes
[[[Pass Station [<sorts.radar.system.station.TX object at 0x7f326b897b40> ...  Rise 8:20:05.086734 (3.5 min) 8:23:35.792890 Fall]]]

If we now look at the number of states within the passes, we get:

>>> n = 0
>>> for pi in radar_passes[0][0]:
...     n += len(pi.inds)
>>> n
772

Which represents a **reduction of 97.2% of the number of states**. Finally, the passes can be plotted by running:

>>> fig = plt.figure()
>>> ax = fig.add_subplot(111, projection='3d')
>>> sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)
>>> for tx in radar.tx:
...     ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
>>> for rx in radar.rx:
...     ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')
>>> ax.plot(object_states[0], object_states[1], object_states[2], '--b', alpha=0.15)
>>> for pi in radar_passes[0][0]:
...     ax.plot(object_states[0, pi.inds], object_states[1, pi.inds], object_states[2, pi.inds], '-r')
>>> plt.show()

.. figure:: ../../../figures/radar_passes.png
	:scale: 75%
