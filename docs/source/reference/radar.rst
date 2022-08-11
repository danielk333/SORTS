.. currentmodule:: sorts.radar.system
.. _radar_system:

=============
Radar systems
=============

Definition
----------
A radar (`RAdio Detection And Ranging`) uses radio waves for the detection of a target and for the estimation of its intrinsinc/dynamic properties :

   * Range
   * Azimuth and elevation
   * Radial velocity (range rate)
   

The ``SORTS`` Toolbox provides a set of features for the definition and simulation of custom radar systems. The current implemenation of ``SORTS`` also includes a set of predefined radar systems which can easily be implemented within your project. 


Role of the radar system
------------------------
The radar system is one of the three central components of the SORTS library. It manages all the 


Structure of Radar system
-------------------------
The ``SORTS`` Toolbox mimics the fundamental architecture of radar systems which are often comprised of two main subsystems :

    * A ``Transmitter`` (or :class:`TX<station.TX>` station) : its role is to transmit electromagnetic waves in the microwave domain in the direction of the target.
    * A ``Receiver`` (or :class:`TX<station.TX>` station) which aquires the portion of the signal scattered by the target. Radar receiver are also in charge of the processing of the input signal (amplification, mixing, analog to digital conversion, ...).

The choice to model the real-world structure of a radar system while ensuring at the same time the highest level of modulability comes at the cost of slightly lower computational performances. But the user-centric design of this toolbox allows for an intuitive use of this library over a wide set of applications.

The SORTS radar system implementation relies on two separate modules : 

   .. toctree::
      :maxdepth: 1

      radar/system/radar
      radar/system/station     


Radar related simulation modules
--------------------------------
SORTS includes a set of radar-related simulation modules to help simulate radar signal propagation, measurements, measurement errors, scanning schemes... Those modules include a set of predifined instances, but also offer the user with a well-defined structure to implement new solutions which satisfy its requirements.

   .. toctree::
      :maxdepth: 1

      radar/scans/scans_base
      radar/signals/signals_base
      radar/passes/passes_base
      radar/measurements/measurements_base
      radar/measurement_errors/measurement_errors_base



Example
-------
To define a radar system, it is possible to use some of the :ref:`predifined radar instances<predifined_radar_instances>` which can be directly imported from the ``sorts.radars`` module.


