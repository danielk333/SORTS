#!/usr/bin/env python

'''
==========================
Atmospheric drag variation
==========================

Showcases the use of the ``sorts.propagation_errors`` module to estimate propagation errors 
due to atmospheric drag. 
'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import sorts

# definition of the space object
obj = sorts.SpaceObject(
    sorts.targets.propagator.SGP4,
    propagator_options = dict(
        settings=dict(
            in_frame='GCRS',
            out_frame='ITRS',
        ),
    ),
    a = 6800e3, 
    e = 0.0, 
    i = 69, 
    raan = 0, 
    aop = 0, 
    mu0 = 0, 
    epoch = Time(57125.7729, format='mjd'),
    parameters = dict(
        A = 2.0,
    )
)

# compute and plot atmospheric drag errors
hour0, offset, t1, alpha = sorts.propagation_errors.atmospheric_drag.atmospheric_errors(obj, plot=True)
plt.show()