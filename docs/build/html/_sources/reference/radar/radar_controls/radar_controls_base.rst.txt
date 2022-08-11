.. currentmodule:: sorts.radar.radar_controls

.. _radar_controls:

==============
Radar controls
==============

1. Description
--------------
**Radar controls** define the sequence of planned transitions :ref:`radar system <radar>` states :math:`(t^k, \mathbf{x}_{rad}^k)` required for the radar to achieve a given objective (tracking of an object, scanning, increasing the power to a given value, ...). 

The state of a :ref:`radar system <radar>` :math:`\mathbf{x}_{rad}^k` at time :math:`t^k` corresponds to the union of the states of all :ref:`stations <station>` comprising the :ref:`radar system <radar>`, such that 

.. math:: 		 \mathbf{x}_{rad}^k = \{ \mathbf{x}_{station, i}^k | i \in [\![ 0, n_{stations}-1 ]\!] \}

The duration of a control sequence is called the **control interval**.


1.1 Control time slices
~~~~~~~~~~~~~~~~~~~~~~~
Each radar control :math:`\mathbf{x}_{rad}^k` is associated with a time interval :math:`[t^k, t^k + \Delta t_k]` during which this control is considered to be active. This time interval is called the **control time slice**. In SORTS, a time slice is defined by two parameters:

 * The time slice **start time** :math:`t^k`. 
 * The time slice **duration** :math:`\Delta t^k`

Since most time slice durations are very small compared to the control interval, it is common to assimilate time slices to their start time :math:`t^k`. In that case, time slices are considered as points (often called **control time points** within SORTS documentation), but it is essential remember that in reality, time slices correspond to time intervals.

.. note::
	In SORTS, time is **continuous** and the start time / duration of a time slice can take any real positive value.

1.2 Controls
~~~~~~~~~~~~
As stated before, a radar control corresponds to a planned transition in the state of a :ref:`radar system <radar>` :math:`\mathbf{x}_{rad}^k`. Consider for example a **radar station property** :math:`x_{ij}` (:math:`j^{th}` property of the :math:`i^{th}` radar station). The control sequence associated with this property will therefore be the list of values :math:`x_{ij}^k` at each time slice :math:`t^k` (under the **"control time point"** approximation).

The current implementation of SORTS supports two types of controls:
 * Pointing directions (ECEF coordinate frame) :
 	The *station pointing direction* corresponds to the normalized Poynting vector of the transmitted signal (i.e. its direction of propagation / direction of the beam). It is possible to have multiple pointing direction vectors for a sinle station per time slice. This feature can therefore be used to model digital beam steering of phased array receiver antennas (see :ref:`radar_eiscat3d`).
 * Station properties :
 	The *station properties* of a radar system correspond to the controllable states of a radar system other than its orientation (number of IPP per time slice, transmit power, pulse length, wavelength, ...). For the sake of optimization (vectorization), the default implementation of SORTS **only supports scalar property values**. 

1.3 Control periods
~~~~~~~~~~~~~~~~~~~
For optimization purposes, computations relative to radar controls are vectorized. Despite the undeniable improvements in computation speed, vectorization has a great impact over RAM usage. To ensure the stability of the toolbox for long simulation times (where single control arrays can easily reach millions of elements), SORTS splits control arrays into multiple **control periods**.

Computations are vectorized within a control period to maximize performances, and each computation result is stored into a separate sub-array. As an example, the control time array (containing the start time of each control time slice) is outputed as follow:

>>> controls.t -> [[control period 1], [control period 2], .... , [control period N-1]]

And each sub-array contains the time points within the given control period:

>>> controls.t[0]
array([0., 1., 2., 3., 4.]) 
>>> controls.t[1]
array([5., 6., 7., 8., 9.])

All the controls (pointing directions and station properties) are also splitted according to those control periods. 

The control period can be defined in two different ways :

 * By setting the **maximum number of points** ``max_points`` within each control period, the controls will be generated such that each control period contains ``max_points`` time points for ``period_id<n_periods-1`` and less than ``max_points`` time points for ``period_id == n_periods-1``.
 * By using a **scheduler** (defined by its starting point ``t0`` and duration ``scheduler_period``), all the  will be generated such that all control periods coincide with the scheduler periods. Beware that while the controller and scheduler periods will correspond to the sime time intervals, the index of the controller period in the control arrays will be different from the corresponding scheduler period if the start time of the scheduler ``t0`` is different from the start time of the controller. 

.. note::
	When using multiple radar controls, the control periods must be generated using the second method (scheduler) to ensure time synchronization between each the periods of different controls0

.. figure:: ../../../figures/example_control_periods.png
	:width: 85%

	Example of the slicing of a control interval using the scheduler control periods. The dashed transition lines are common to both the scheduler and the controller, but the period indices are different.

1.4 Radar states
----------------
In SORTS, :class:`radar controls<RadarControls>` are also used to model the states of the radar system in time after being controlled. Those radar states can be obtained by running the :attr:`control<sorts.radar.system.radar.Radar.control>` method of the :class:`Radar<sorts.radar.system.radar.Radar>` class using a given control sequence:

>>> radar_states = radar.control(control_sequence)

The resulting ``radar_states`` variable will be an instance of :class:`radar controls<RadarControls>` containing all the actual consecutive states of the radar when running the control sequence ``control_sequence``. If some variables are not controlled by ``control_sequence``, ``radar_states`` will contain the default values of the radar properties of each radar :class:`station<sorts.radar.system.station.Station>` 

2. API Reference
----------------
The *Radar controls* component allows users to easily store and manage control sequences using a common object : the :class:`sorts.RadarControls<RadarControls>` class. 

.. autosummary::
	:toctree: auto/

	~RadarControls
