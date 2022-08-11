.. currentmodule:: sorts.radar

.. _controllers_and_schedulers:

================================
Radar Controllers and Schedulers
================================

1. Description
--------------
:class:`Station properties<sorts.radar.system.station.Station.PROPERTIES>` (such as the wavelength of the signal, power, the azimuth and elevation of the antenna, ...) must be well defined in order to perform radar observations of :class:`space objects<sorts.targets.space_object.SpaceObjects>`. Such properties are often controlled by a set of commands (which in sorts is called a **radar control sequence**) which values can vary in time during the measurement interval. 

In SORTS, simulations of radar observations require for the radar to be controlled during the entirety of the measurement interval. Controlling a radar system ensures that the state of the radar is well defined during each measurement simulation step. The structure of the main control-related components of sorts during the simulation of observations is the following:

.. figure:: ../figures/diagram_controllers_and_schedulers.png

The role of :ref:`radar controllers<controllers>` is to generate **radar control sequences** which can be executed by the :ref:`radar system<radar>` to perform a set of predifined actions (tracking of a space object, scanning of an area, ...). When multiple control sequences are overlapping over a common time interval, a :ref:`radar scheduler<scheduler>` is used to generate a new control sequence. Schedulers combine a set of multiple control sequences into a single one by satisfying a set of constraints given by the type of scheduler used.


2. Radar control related modules
--------------------------------
As seen earlier, SORTS radar control features are carried out by three main modules:

.. toctree::
	:maxdepth: 1

	radar/radar_controls/radar_controls_base
	radar/controllers/controllers_base
	radar/scheduler/scheduler_base

Each one of those modules is designed to ensure extensibility and computation efficiency.