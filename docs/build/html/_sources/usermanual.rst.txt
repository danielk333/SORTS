.. _reference:

############
SORTS Manual
############

:Release: |version|
:Date: |today|

Introduction
------------
The **SORTS Toolbox** was created to provide researchers and engineers a complete and easy-to-use library to simulate radar observations of **Near-Earth space Objects** (NEO) by providing a common interface for the three main components involved in the simulation of radar measurements :

* **Radars :** radar systems are the central component of the observational system as it manages all the aspects related to the pointing and properties of the high-power signal, its aquisition and processing and many more. As such, the SORTS toolbox provides a set of functions to simulate the behaviour, performances and measurements of radar systems.
* **Targets :** SORTS allows for the simulation and handling of a population of space objects undergoing the radar measurements. Amongst the many features provided, SORTS wraps a set of well-known orbital state propagators together with easy-to-use interfaces with standard space object databases (Master catalog, TLE, ...).
* **Radar controllers and schedulers :** the generation of basic and well-defined instructions for the radar system is necessary to conduct observations of space objects. Wether the objective is to track space objects or to perform Beam-Park experiments, SORTS allows for the generation and scheduling radar controls. 

The *SORTS* library also include a wide set of additional modules which can be used to perform large scale simulations, error estimates of measurements, plot simulation results, perform coordinate transformations and many more. 


Summary
-------
The purpose of this guide is to provide a complete description of the structure of the toolbox. It also provides the reader with a set of examples together with a in-depth documentation of the functions, modules and classes : 

.. toctree::
   :maxdepth: 1

   reference/radar
   reference/targets
   reference/controllers_and_schedulers
   reference/io
   reference/transformations
   reference/plotting
   reference/correlator
   reference/common
   reference/c-api
