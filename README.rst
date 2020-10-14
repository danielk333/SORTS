SORTS
=========


Feature list
-------------

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
* All time-critical calculations implemented using numpy to accelerate calculation
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


Install SORTS
-----------------

**To install SORTS**

.. code-block:: bash

   git clone https://github.com/danielk333/SORTS
   cd sorts
   pip install .

The installation can be automatically tested if `pytest` is also installed

.. code-block:: bash

   pytest


From scratch
---------------

**Make sure you have Python >= 3.6**

To install Python 3.7 (from the deadsnakes ppa) on Ubuntu 16 (WARNING: This method differs on Ubuntu 18/20)

.. code-block:: bash

   sudo apt update
   sudo apt install software-properties-common
   sudo add-apt-repository ppa:deadsnakes/ppa
   sudo apt update
   sudo apt install python3.7
   sudo apt install python3.7-venv

It is recommended to use virtual environments when installing `sorts`

.. code-block:: bash

   python3.7 -m venv /path/to/new/sorts/environment
   source /path/to/new/sorts/environment/bin/activate

Now you should be inside the new virtual environment. Check this by

.. code-block:: bash

   pip --version

And you should see the path to "/path/to/new/sorts/environment". 

Alternatively if you want to use pip with Python 3.7 without using `venv`, the get-pip.py method can be used. WARNING: Using the bootstrap solution will break your current Python 2.7/3.5 pip, It is not recommended.

.. code-block:: bash
   sudp apt install curl
   curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
   python3.7 get-pip.py --user


Now, regardless of the method, you should make sure you have an up to date pip:

.. code-block:: bash

   pip install --upgrade pip


If you do not have git, install it first:

.. code-block:: bash

   sudo apt install git


**To install SORTS**

.. code-block:: bash

   git clone https://github.com/danielk333/SORTS
   cd sorts
   pip install .

In case "pyant" or "pyorb" requirements fail on auto-install, run the following commands manually and try again:

.. code-block:: bash

   pip install git+https://github.com/danielk333/pyant
   pip install git+https://github.com/danielk333/pyorb

If you have trouble getting plotts from the examples, you might need to install a GUI-backed to `matplotlib` like TkAgg

.. code-block:: bash

   sudo apt install python3.7-tk

and force matplotlib to use that backend by creating a matplotlibrc file ( https://matplotlib.org/tutorials/introductory/customizing.html ) and adding "backend : TkAgg" to it.


Install MPI
--------------

Open MPI on Ubuntu

.. code-block:: bash

   sudo apt update
   sudo apt install openmpi-bin libopenmpi-dev
   pip install mpi4py


MPICH on Ubuntu

.. code-block:: bash

   sudo apt-get update
   sudo apt install mpich
   pip install mpi4py

Install Orekit
----------------

Using install script while a virtual environment is active on Ubuntu (from inside the SORTS repository)

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install openjdk-8-jdk
   ./install_orekit.sh


Install Pyglow
---------------

Taken from "https://github.com/timduly4/pyglow/"

.. code-block:: bash

  git clone git://github.com/timduly4/pyglow.git pyglow

  cd pyglow/
  pip install -r requirements.txt
  make -C src/pyglow/models source
  python setup.py install


Example
---------------

Finding passes over radar system

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
      max_dpos=1e3,
  )

  states = prop.propagate(t, orb.cartesian[:,0], orb.epoch)

  passes = eiscat3d.find_passes(t, states)

  print(passes)


For developers
===============

To install developer dependencies 
------------------------------------

.. code-block:: bash

   #NOT YET AVALIBLE


To test
-----------------

.. code-block:: bash

   pytest



To make doc
-----------------

To compile the github pages documentation run

.. code-block:: bash

   git checkout gh-pages
   git cd docsrc
   make github

Otherwise, one can compile the documentation directly on the current branch by running 

.. code-block:: bash

   git cd docsrc
   make html

which causes the output to go into the "build" folder.

When used for publications
===========================

A paper and a DOI is underway and will soon be available, for now: please just tell us by email (daniel.kastinen@irf.se) or here on Github.

@article{
    autor="",
    title=""
}

