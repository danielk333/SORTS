SORTS
=========


Feature list
-------------

* TO BE ADDED


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

and force matplotlib to use that backend by creating a matplotlibrc file ( https://matplotlib.org/tutorials/introductory/customizing.html ) and adding `backend : TkAgg` to it.


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

**THIS MAKE TARGET NEEDS UPDATE** (currently manually handled with `make html` instead)

.. code-block:: bash

   git checkout gh-pages
   git cd docsrc
   make github



When used for publications
===========================

@article{
    autor="",
    title=""
}

