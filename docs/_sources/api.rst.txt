==============
API Reference
==============

Modules
========

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts

    sorts.space_object
    sorts.passes
    sorts.functions
    sorts.constants
    sorts.frames
    sorts.dates
    sorts.profiling
    sorts.signals


Sub-packages
=============

propagator
-------------

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts/propagator


    sorts.propagator.base
    sorts.propagator.orekit
    sorts.propagator.pysgp4


population
-------------

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts/population


    sorts.population.population
    sorts.population.master


controller
-------------

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts/controller


    sorts.controller.radar_controller
    sorts.controller.scanner
    sorts.controller.tracker



scheduler
-------------

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts/scheduler


    sorts.scheduler.scheduler
    sorts.scheduler.observed_parameters
    sorts.scheduler.static_list
    sorts.scheduler.tracking



errors
-------------


io
-------------


plotting
-------------


radar
-------------



scans
-------------

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts/scheduler


    sorts.radar.scans.scan
    sorts.radar.scans.uniform
    sorts.radar.scans.random_uniform
    sorts.radar.scans.fence
    sorts.radar.scans.plane


Instances
=========

Radars
-------

.. automodule:: sorts.radar.instances

