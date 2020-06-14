#!/usr/bin/env python

'''Example showing how SGP4 propagator can be used

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyorb

from sorts.propagator import SGP4

prop = SGP4(
    settings = dict(
        out_frame='TEME',
    ),
)

print(prop)

orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees = True, a=7000e3, e=0, i=69, omega=0, Omega=0, anom=0)
print(orb)

t = np.linspace(0,3600*24.0*15,num=5000)
mjd0 = 53005

states = prop.propagate(t, orb.cartesian[:,0], mjd0, A=1.0, C_R = 1.0, C_D = 1.0)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0,:], states[1,:], states[2,:],"-b")

plt.show()