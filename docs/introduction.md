# Introduction

Space Object Radar Tracking Simulator (SORTS) is a python library for radar simulation of space objects.

## Simplified Architecture

The conceptual core of this library is quite simple. Below is a simplified architecture of a simulation using `sorts`:

![diagram](/assets/simplified_simulation_architecture.svg)

It is helpful to first develop this simplified conceptual understanding.
In practice, the function signatures are more complex, and preparing
the input parameters can obscure the overall structure. However, the
underlying idea remains the same.

This architecture will be explained in more detail in the [tutorial](/tutorial/tutorial/).


## Subpackages and Modules

The library organizes the different functions and stages of a radar simulation into subpackages, allowing easy composition and customization of simulations.

The main subpackages are grouped as follows:

- Representaions
    - `space_object`: Represents a space object.
    - `population`: Represents a distribution of space object.
    - `radar`: Represents a radar system.
- Signal processing
    - `signals`: Radar singal calculations.
- Simulation
    - `schedule`: Handles resolution of multiple schedules
    - `passage`: Represents the passover of a space object over the field of view of a set of radar stations.
    - `pointing`: Generates schedules which controls the pointings of a radar system.
    - `simulation`: Handles simulation.
- Physics utilities
    - `frames`: Handles the various transformation between coordinate systems.
    - `interpolation`: Handles the interpolation of a space object motion states.
- Other utilities
    - `plotting`: Collection of plotting helpers and functions.
    - `mpi_job_queue_executor`: Handles parallelzied exection via Message Passing Interface (MPI).

## Functionality Highlights

- Vast library of usage examples
- Quick calculation of passes over a radar system
- Easy simulation of observed variables of hard targets with radar systems (range, range rate, radar cross section, signal to noise ratio, ...) given an arbitrary radar and radar observation schema
- Definition of arbitrary radar control systems
- Definition of arbitrary scheduler systems that manage radar-controllers
- Pre-defined library of radar systems, radar survay patterns, standard radar controllers and schedulers
- Standardized interface to a collection of propagators
- Allows modification of any level of a simulation trough sub-classing the basic models
- Large collection of helper functions for simulation to automate e.g. MPI-trivial parallelization and disk-caching
- Execution time and memory usage profiler compatible with most base models
- Logging compatible with most base models
- Frame transformations implemented trough Astropy
- Time handling implemented trough Astropy
- All time-critical calculations implemented using numpy to accelerate calculation
- Predefined error models such as ionospheric ray bending and coded transmission matched filter errors
- Ray-tracing simulation of radar signals trough the ionosphere using pyglow
- Measurement Jacobian calculation and linearized orbit error calculation and propagation
- Ability to plan measurement campaigns using the output from a scheduler simulation
- Collection of predefined population formats for loading e.g. TLE catalogs
- Correlation algorithms for correlating measurement data to a population
- Input/Output package for writing and reading standardized data formats e.g. CCSDS TDM files
- Large collection of plotting functions for quick visualization
- Interpolation methods for propagation optimization
- ...
