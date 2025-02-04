==============
API Reference
==============

clibsorts
===========

.. c:autodoc:: clibsorts/measurements.c
   :transform: napoleon

Modules
========

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts

    sorts.constants
    sorts.correlator
    sorts.dates
    sorts.frames
    sorts.functions
    sorts.interpolation
    sorts.passes
    sorts.profiling
    sorts.signals
    sorts.simulation
    sorts.space_object


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
    sorts.population.tles


controller
-------------

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts/controller


    sorts.controller.radar_controller
    sorts.controller.scanner
    sorts.controller.tracker
    sorts.controller.static



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

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts/errors


    sorts.errors.errors
    sorts.errors.ionospheric_ray_trace
    sorts.errors.linearized_coded
    sorts.errors.atmospheric_drag
    sorts.errors.linearized_orbit_determination


io
-------------

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts/io


    sorts.io.ccsds


plotting
-------------

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts/plotting


    sorts.plotting.general
    sorts.plotting.radar
    sorts.plotting.scan
    sorts.plotting.tracking



radar
-------------

.. autosummary::
   :template: autosummary/module.rst
   :toctree: _autodoc/sorts/radar


    sorts.radar.radar
    sorts.radar.tx_rx


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
    sorts.radar.scans.bp



Instances
=========

Radars
-------

.. automodule:: sorts.radar.instances
