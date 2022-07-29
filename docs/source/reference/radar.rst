.. currentmodule:: sorts.radar.system
.. _radar:

=============
Radar systems
=============

Definition
----------
A radar (`RAdio Detection And Ranging`) system is a detection system which uses radio waves for the determination of a target :

   * range
   * azimuth and elevation
   * radial velocity

``SORTS`` provides a set of classes and functions to define and simulate custom radar systems. It also includes a set of already defined Radar instances which can easily used within your project.


Structure of Radar system
-------------------------
A radar system can be decomposed into two main subsystems :

    * A ``transmitter`` (or :class:`TX<station.TX>` station) which role is to transmit pulses of electromagnetic radiations in the microwave domain in the direction of the target which we want to characterize.
    * A ``receiver`` (or :class:`TX<station.TX>` station) which aquires the portion of the signal scattered by the target. The receiver is often also in charge of processing the input signal (amplification, mixing, analog to digital conversion, ...).


To mimic the real-world structure of a radar system, the SORTS radar system implementation relies on two separate modules : 

   .. toctree::
      :maxdepth: 1

      radar/system/radar
      radar/system/station      

Examples
--------
The SORTS radar system have been designed to mimic the structure of real radar systems. This implementation allows for a better understanding of the role of each module, but also for a more intuitive use of the library functionalities.

..  