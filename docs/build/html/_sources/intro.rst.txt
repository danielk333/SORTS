.. _sorts_intro

############
Introduction
############

What is SORTS ?
---------------
The Space Object Radar Tracking Simulator (SORTS) toolbox is a collection of modules designed for research purposes concerning the tracking and detection of objects in space. Its ultimate goal is to simulate the tracking and discovery of space objects using radar systems in a very general fashion. Therefore, it can not only be used for simulating the performance of radar systems, but also for the plannning and scheduleing of observation campagins.



Features
--------
SORTS includes the following set of features : 

* Vast library of usage examples
* Quick calculation of passes over a radar system
* Easy simulation of observed variables of hard targets with radar systems (range, range rate, radar cross section, signal to noise ratio, ...) given an arbitrary radar and radar observation schema
* Definition of arbitrary radar control systems
* Definition of arbitrary scheduler systems that manage radar-controllers
* Pre-defined library of radar systems, radar survay patterns, standard radar controllers and schedulers
* Standardized interface to a collection of propagators
* Allows modification of any level of a simulation trough sub-classing the basic models
* Large collection of helper functions for simulation to automate e.g. MPI-trivial parallelization and disk-caching
* Execution time and memory usage profiler compatible with most base models
* Logging compatible with most base models
* Frame transformations implemented trough Astropy
* Time handling implemented trough Astropy
* All time-critical calculations implemented using numpy & C to accelerate calculation
* Predefined error models such as ionospheric ray bending and coded transmission matched filter errors
* Ray-tracing simulation of radar signals trough the ionosphere using pyglow
* Measurement Jacobian calculation and linearized orbit error calculation and propagation
* Ability to plan measurement campaigns using the output from a scheduler simulation
* Collection of predefined population formats for loading e.g. TLE catalogs
* Correlation algorithms for correlating measurement data to a population
* Input/Output package for writing and reading standardized data formats e.g. CCSDS TDM files
* Large collection of plotting functions for quick visualization 
* Interpolation methods for propagation optimization
* ...


Example
-------
A simple application of SORTS consists in finding all passes of a space object over a radar system :

.. code-block:: python

    #!/usr/bin/env python

    import numpy as np
    import pyorb

    import sorts
    from sorts.propagator import SGP4

    eiscat3d = sorts.radars.eiscat3d

    prop = SGP4(
        settings = dict(
            out_frame='ITRS',
        ),
    )

    orb = pyorb.Orbit(
        M0 = pyorb.M_earth, 
        direct_update=True, 
        auto_update=True, 
        degrees=True, 
        a=7200e3, 
        e=0.05, 
        i=75, 
        omega=0, 
        Omega=79, 
        anom=72, 
        epoch=53005.0,
    )
    print(orb)

    t = sorts.equidistant_sampling(
        orbit = orb, 
        start_t = 0, 
        end_t = 3600*24*1, 
        max_dpos=1e4,
    )

    states = prop.propagate(t, orb.cartesian[:,0], orb.epoch)

    passes = eiscat3d.find_passes(t, states)

    for txi in range(len(eiscat3d.tx)):
        for rxi in range(len(eiscat3d.rx)):
            for ps in passes[txi][rxi]: print(ps)