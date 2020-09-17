Introduction
=================

What is SORTS
-----------------
SORTS stands for Space Object Radar Tracking Simulator (SORTS). It is a collection of modules designed for research purposes concerning the tracking and detection of objects in space. Its ultimate goal is to simulate the tracking and discovery of objects in space using radar systems in a very general fashion. Therefor it can not only be used for simulating the performance of radar systems but be used to plan observations and schedule campagins.



Feature list
-------------

* TO BE ADDED




Example
----------

Finding all passes of a space object over a radar system

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


