Introduction
===================================

What is SORTS
-----------------
SORTS stands for Space Object Radar Tracking Simulator (SORTS). It is a collection of modules designed for research purposes concerning the tracking and detection of objects in space. Its ultimate goal is to simulate the tracking and discovery of objects in space using radar systems in a very general fashion. Therefor it can not only be used for simulating the performance of radar systems but be used to plan observations and schedule campagins.


Install
-----------------

System requirements
~~~~~~~~~~~~~~~~~~~~~~

* Unix (tested on Ubuntu-16.04 LTS, Ubuntu-server-16.04 LTS)
* Python > 3.5

Dependencies
~~~~~~~~~~~~~

.. include:: ../../../requirements
   :literal:



First Simulation
~~~~~~~~~~~~~~~~~~

**Note**: To run the simulation with MPI the file must be executable.

.. code-block:: bash

   python ./examples/getting_started.py

or with 

.. code-block:: bash

   mpirun -np 8 ./examples/getting_started.py

if you wish to test the MPI implementation of the simulation. The *-np* specifies how many processes should be launched and should not be larger then the number of cores available.

License
------------------

.. include:: ../../../LICENSE

