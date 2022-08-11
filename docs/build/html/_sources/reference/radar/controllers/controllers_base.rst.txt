.. currentmodule:: sorts.radar.controllers

.. _controllers:

=================
Radar controllers
=================

Description
-----------
:ref:`Radar controllers<controllers>` are used for the generation of radar control sequences which, when applied to a radar system, allow it to perform certain types of actions (tracking of space objects, scanning schemes, …).

The design of the radar controller allows the user to generate multiple controls of the same type with only one controller instance.

 .. admonition:: Example

	A single instance of the :class:`Scanner<scanner.Scanner>` controller can be used to generate any scanning control sequences for any of the scan types available within SORTS. 

When generating a new control sequence, the controller must define the following characteristics: 
 * The **control interval** :math:`[t_i, t_f]` during which the control is active.
 * The **start time** :math:`\Delta t^k` of each time slice.
 * The **duration** :math:`\Delta t^k` of each time slice.
 * The value :math:`x_i^k` of each **station property** being controlled for every time slice. 
 * The **pointing direction** :math:`\mathbf{p}_i^k` of each station for each time slice.


API Reference
-------------

1. Controller Base Class
~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`sorts.RadarController<radar_controller.RadarController>` class defines the fundamental architecture of a radar controller.

.. autosummary::
	:toctree: auto/

	~radar_controller.RadarController

2. Predifined controllers
~~~~~~~~~~~~~~~~~~~~~~~~~
SORTS implementation contains a set of predifined :class:`controller<radar_controller.RadarController>` instance.

.. autosummary::
	:toctree: auto/

	~scanner.Scanner
	~tracker.Tracker
	~static.Static
	~space_object_tracker.SpaceObjectTracker