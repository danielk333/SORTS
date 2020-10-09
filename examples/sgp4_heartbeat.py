#!/usr/bin/env python

'''
Using the heartbeat if implemented
========================================
'''

import numpy as np
import pyorb
from astropy.time import Time

from sorts.propagator import SGP4

#it will run a bit slower since it cant use the "array" optimization of SGP4

class MySGP4(SGP4):
    def heartbeat(self, t, state, satellite):
        print(f'We are now looking at a satellite {satellite.satnum} at {t}')


prop = MySGP4(
    settings = dict(
        out_frame='TEME',
        heartbeat=True,
    ),
)
orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees = True, a=7000e3, e=0, i=69, omega=0, Omega=0, anom=0)
print(orb)

t = np.linspace(0,3600*24.0,num=10)

x = prop.propagate(
    t, 
    orb.cartesian[:,0], 
    epoch=Time(53005, format='mjd', scale='utc'), 
    A=1.0, 
    A_std = 0.1,
    relative_variation = True,
    C_R = 1.0,
    C_D = 1.0,
)

#stop the heartbeat
prop.set(heartbeat=False)
x = prop.propagate(
    t, 
    orb.cartesian[:,0], 
    epoch=Time(53005, format='mjd', scale='utc'), 
    A=1.0, 
    A_std = 0.1,
    relative_variation = True,
    C_R = 1.0,
    C_D = 1.0,
)
