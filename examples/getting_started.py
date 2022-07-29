#!/usr/bin/env python

'''
================
Starting example
================

This starting example creates an object and lists all the passes over the EISCAT 3D stations
over a time period of 24h.
'''

import numpy as np
import pyorb

import sorts
from sorts.targets.propagator import SGP4

# create radar
eiscat3d = sorts.radars.eiscat3d

# intializes the SGP4 propagator
prop = SGP4(
    settings = dict(
        out_frame='ITRS',
    ),
)

# intializes the space object orbit
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

# generate time array over a span 24h
t = sorts.equidistant_sampling(
    orbit = orb, 
    start_t = 0, 
    end_t = 3600*24*1, 
    max_dpos=1e4,
)

# propagate space object states
states = prop.propagate(t, orb.cartesian[:,0], orb.epoch)

# gets all passes of the space object over the radar stations
passes = eiscat3d.find_passes(t, states)

# prints passes
for txi in range(len(eiscat3d.tx)):
    for rxi in range(len(eiscat3d.rx)):
        for ps in passes[txi][rxi]: print(ps)