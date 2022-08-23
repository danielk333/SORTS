.. currentmodule:: sorts.targets.propagation_errors

.. _propagation_errors:

==================
Propagation errors
==================

1. Description
--------------
The simplifications used to implement the propagators are often important sources of errors which can impact the precision of the simulation. The :ref:`propagation_errors` module provides a set of functions to estimate the errors in position, time and velocity due to different sources of errors (atmospheric drag, ...).

2. API Reference
----------------

1.1 Atmospheric drag
~~~~~~~~~~~~~~~~~~~~

.. autosummary::
	:toctree: auto/

	~atmospheric_drag