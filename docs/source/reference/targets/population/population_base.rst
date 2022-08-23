.. currentmodule:: sorts.targets.population

.. _population:

==========
Population
==========

1. Description
--------------
Populations of space object can be used to store and manage sets of space objects. The :ref:`population` module offers a set of functions and classes which can be used to interface with standard Near-Earth space object catalogs and formats (TLE, Master catalog, ...).

The population class acts as a specialized dictionnary where the parameters of space objects as defined by the user/catalogs can be stored and modified. Those parameters can then be used to generate :class:`Space objects<sorts.targets.space_object.SpaceObject>` according to the parameters stored within the population.

2. API Reference
----------------
2.1. The population class
~~~~~~~~~~~~~~~~~~~~~~~~~
The current implementation of SORTS defines the base population class.

.. autosummary::
	:toctree: auto/

	~base.Population

2.2. Catalog interfaces
~~~~~~~~~~~~~~~~~~~~~~~
Catalog modules can be used as import/export tools to interact with standardized space object catalogs.

2.2.1. Master catalog
^^^^^^^^^^^^^^^^^^^^^
.. autosummary:: 
	:toctree: auto/

	~master

2.2.2. TLE catalog
^^^^^^^^^^^^^^^^^^
.. autosummary:: 
	:toctree: auto/

	~tles