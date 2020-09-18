SORTS
=========


Feature list
-------------

* TO BE ADDED


Install SORTS
-----------------

**Make sure you have Python >= 3.6**

It is recommended to use virtual environments

.. code-block:: bash

   python3 -m venv /path/to/new/sorts/environment
   source /path/to/new/sorts/environment/bin/activate

Now you should be inside the new virtual environment. Check this by

.. code-block:: bash

   pip --version

And you should see the path to "/path/to/new/sorts/environment". If the "pip" command fails, you need to make sure you have a functioning Python version with pip.

Now you should make sure you have an up to date pip:

.. code-block:: bash

   pip install --upgrade pip

**To install SORTS**

.. code-block:: bash

   git clone https://github.com/danielk333/SORTS
   cd sorts
   pip install .

In case "pyant" or "pyorb" requrements fail auto-install run the following commands:

.. code-block:: bash

   pip install git+https://github.com/danielk333/pyant
   pip install git+https://github.com/danielk333/pyorb


Install MPI
--------------

Open MPI on Ubuntu

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install openmpi-bin libopenmpi-dev
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

