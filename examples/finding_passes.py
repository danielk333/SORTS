#!/usr/bin/env python

'''

'''

import numpy as np
import pyorb

import sorts
from sorts.propagator import SGP4
from sorts.radar.instances import eiscat3d

orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees=True, a=7200e3, e=0.1, i=75, omega=0, Omega=79, anom=72, epoch=53005.0)
print(orb)

t = sorts.equidistant_sampling(
    orbit = orb, 
    start_t = 0, 
    end_t = 3600*24, 
    max_dpos=1e4,
)
print(f'Temporal points: {len(t)}')

prop = SGP4(
    settings = dict(
        out_frame='ITRF',
    ),
)

states = prop.propagate(t, orb.cartesian[:,0], orb.epoch, A=1.0, C_R = 1.0, C_D = 1.0)

# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(states[0,:], states[1,:], states[2,:],"-b")

# plt.show()

passes = sorts.find_passes(t, states, eiscat3d)