.. currentmodule:: sorts.radar.scheduler

.. _schedulers:

================
Radar schedulers
================

Description
-----------
Radar schedulers are used to combine multiple (possibly conflicting) types of radar controls. Its goal is to generate a new control sequence which will contain a portion of the time slices of the input controls according to a set of criteria specific to the type of scheduler. 

The research of new schedulers (optimizing the management of radar ressources) being an open research topic, SORTS defines a standard scheduler structure which can be used to define new scheduler types and simulate their results.  

API Reference
-------------
1. Scheduler Base Class
~~~~~~~~~~~~~~~~~~~~~~~
The :class:`sorts.RadarSchedulerBase<base.RadarSchedulerBase>` class defines the fundamental architecture of a radar scheduler.

.. autosummary::
	:toctree: auto/

	~base.RadarSchedulerBase

2. Predifined schedulers
~~~~~~~~~~~~~~~~~~~~~~~~
The current implementation of SORTS includes a single predifined :class:`scheduler<base.RadarSchedulerBase>` instance.

.. autosummary::
	:toctree: auto/

	~static_priority_scheduler.StaticPriorityScheduler