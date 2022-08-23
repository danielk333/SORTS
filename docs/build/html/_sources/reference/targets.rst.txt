.. currentmodule:: sorts.targets
.. _targets:

=============
Radar targets
=============

1. Description
--------------
SORTS simulates the dynamics of space objects located close to earth. The simulation relies on two main components :
 - The :class:`Space Object<space_object.SpaceObject>` which encapsulates a space object in orbit around Earth.
 - The :class:`Propagator<propagator.base.Propagator>` which simulates the dynamics of the :class:`Space Object<space_object.SpaceObject>`.

2. Simulation of the dynamics
-----------------------------
The state of a space object can be described by a 6D state vector

.. math:: \mathbf{x}(t) = [x(t), y(t), z(t), v_x(t), v_y(t), v_z(t)]^T

The role of the :class:`Propagator<propagator.base.Propagator>` is to solve the differential equations governing the dynamics of the space object. Given the complexity of multibody interactions, it is often possible (and required) to perform a set of simplifications to ensure rapid computations (regarding inter-body interactions, Atmospheric and Radiation pressure Drag, gravity model, ...). Different :class:`Propagator<propagator.base.Propagator>` are built upon their own set of simplifications and must be chosen according to the precision expected and the configuration of the problem.

In SORTS, a space object is modelled as a point mass in orbit around the Earth. The :class:`Space Object<space_object.SpaceObject>` allows for the definition of other parameters (such as atmospheric drag coefficient, cross section, ...) which can be used to simulate radar observations or to propagate states.

3. API Reference
----------------
3.1. The :class:`Space Object<space_object.SpaceObject>` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`Space Object<space_object.SpaceObject>` class encapsulates a space object in orbit around the Earth. It contains all the information needed for the simulation of its dynamics and observation by radar systems.

.. toctree::

	targets/space_object/space_object_base

3.2. The :class:`Propagator<propagator.base.Propagator>` class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The :class:`Propagator<propagator.base.Propagator>` class wraps multiple propagation libraries which can be used to compute the states of space objects as a function of time in multiple reference frames.

.. toctree::

	targets/propagator/propagator_base

3.3. Populations of space objects
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
:ref:`populations` are used to store and manage sets of space objects (catalogs, subsets, ...). 

.. toctree::

	targets/population/population_base

3.4. Propagation errors
~~~~~~~~~~~~~~~~~~~~~~~
The simplifications used to implement the propagators are often important sources of errors which can impact the precision of the simulation. The :ref:`propagation_errors` module provides a set of functions to estimate the errors in position, time and velocity due to different sources of errors (atmospheric drag, ...).

.. toctree::

	targets/propagation_errors/propagation_errors_base
