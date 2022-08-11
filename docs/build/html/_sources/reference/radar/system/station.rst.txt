.. currentmodule:: sorts.radar.system.station

.. _station:

====================================
Station (sorts.radar.system.station)
====================================

Description
-----------
The station is one of the most basic building blocs of a Radar system. Its role is to handle the emission/reception of the Radar signal, as well as the local signal processing computations.

There are two basic types of stations :
	* The `transmitting` stations (:class:`TX<TX>`) which are responsible for emitting the radar signal. 
	* The `receiving stations` (:class:`RX<RX>`) which are responsible for aquiring the scattered signal and analysing it.

These two types of stations can be controlled in order to achieve predifined scanning schemes or to track space objects which orbits are known (see :ref:`controllers <controllers>`).


.. note::
   If one wants to implement a new type of station, the new class has to inherit from the :class:`Station<Station>` to ensure compatibility with the rest of SORTS library.


Module structure
----------------
The Radar station module contains the definition of 3 classes :

.. autosummary::
   :toctree: station/

   ~Station
   ~TX
   ~RX


(c) 2016-2020 Juha Vierinen, Daniel Kastinen, Thomas Maynadie