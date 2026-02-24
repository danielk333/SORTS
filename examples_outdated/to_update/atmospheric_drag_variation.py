#!/usr/bin/env python

'''
Atmospheric drag variation
================================

'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import sorts


obj = sorts.SpaceObject(
    sorts.propagator.SGP4,
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

hour0, offset, t1, alpha = sorts.errors.atmospheric_drag.atmospheric_errors(obj, plot=True)


plt.show()