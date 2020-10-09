
Installation
===============

SORTS
-------

System requirements
~~~~~~~~~~~~~~~~~~~~~~

* Unix (tested on Ubuntu-16.04 LTS, Ubuntu-server-16.04 LTS)
* Python > 3.6


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

Installation
~~~~~~~~~~~~~~

**To install SORTS**

.. code-block:: bash

   git clone https://github.com/danielk333/SORTS
   cd sorts
   pip install .

In case "pyant" or "pyorb" requrements fail auto-install run the following commands:

.. code-block:: bash

   pip install git+https://github.com/danielk333/pyant
   pip install git+https://github.com/danielk333/pyorb


MPI
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


Orekit
----------------

Using install script while a virtual environment is active on Ubuntu (from inside the SORTS repository)

.. code-block:: bash

   sudo apt-get update
   sudo apt-get install openjdk-8-jdk
   ./install_orekit.sh



For developers
-----------------

To install developer dependencies 

.. code-block:: bash

   #NOT YET AVALIBLE


To test

.. code-block:: bash

   pytest



To make doc

.. code-block:: bash

   git checkout gh-pages
   git cd docsrc
   make github



