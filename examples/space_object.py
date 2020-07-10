
#!/usr/bin/env python

'''
The SpaceObject class
======================================

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sorts.propagator import Orekit
from sorts import SpaceObject

orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'

orekit_options = dict(
    orekit_data = orekit_data, 
    settings=dict(
        in_frame='EME',
        out_frame='EME',
        drag_force = False,
        radiation_pressure = False,
    ),
)

t = np.linspace(0,3600*24.0*2,num=5000)

SO = SpaceObject(
    Orekit,
    propagator_options = orekit_options,
    a = 7000e3, 
    e = 0.0, 
    i = 69, 
    raan = 0, 
    aop = 0, 
    mu0 = 0, 
    mjd0 = 57125.7729
)

print(SO)

states = SO.get_state(t)



fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0,:], states[1,:], states[2,:],"-b")

max_range = np.linalg.norm(states[0:3,0])*2

ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)
plt.show()

