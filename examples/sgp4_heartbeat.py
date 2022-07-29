#!/usr/bin/env python

'''
==================================
Using the heartbeat if implemented
==================================

This example showcases one use case of the ``heartbeat`` option in the SGP4 propagator. When
enabled, this option executes the custom function ``heartbeat(self, t, state, satellite)`` which 
must be defined in the SGP4 class (or in a inherited class).

In this case, the ``heartbeat`` function prints a custom message to keep track of the propagation
status.
'''

import numpy as np
import pyorb
from astropy.time import Time

from sorts.targets.propagator import SGP4

#it will run a bit slower since it cant use the "array" optimization of SGP4
class MySGP4(SGP4):
    def heartbeat(self, t, state, satellite):
        ''' Custom ``heartbeat`` implementation.

        Prints the number of the satellite and the time step at each propagation iteration
        '''
        print(f'We are now looking at a satellite {satellite.satnum} at {t}')

# initialize our custom SGP4 class ``MySGP4`` containing our implementation of the ``heatbeat`` function.
prop = MySGP4(
    settings = dict(
        out_frame='TEME',
        heartbeat=True,
    ),
)
# initializes the orbit state
orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees = True, a=7000e3, e=0, i=69, omega=0, Omega=0, anom=0)
print(orb)

# propagate the states with the heatbeat option enabled
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

# disables the heartbeat option and performs a new propagation
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
