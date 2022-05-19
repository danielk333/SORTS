
#!/usr/bin/env python

'''
The SpaceObject class
======================================

'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time

from sorts.targets.propagator import Orekit
from sorts import SpaceObject
import sorts

try:
    pth = pathlib.Path(__file__).parent.resolve()
except NameError:
    pth = pathlib.Path('.').parent.resolve()
pth = pth / 'data' / 'orekit-data-master.zip'


if not pth.is_file():
    sorts.targets.propagator.Orekit.download_quickstart_data(pth, verbose=True)


orekit_options = dict(
    orekit_data = pth, 
    settings=dict(
        in_frame='GCRS',
        out_frame='GCRS',
        drag_force = False,
        radiation_pressure = False,
    ),
)

t = np.linspace(0,3600*24.0*2,num=5000)

obj = SpaceObject(
    Orekit,
    propagator_options = orekit_options,
    a = 7000e3, 
    e = 0.0, 
    i = 69, 
    raan = 0, 
    aop = 0, 
    mu0 = 0, 
    epoch = Time(57125.7729, format='mjd'),
    parameters = dict(
        d = 0.2,
    )
)

print(obj)

states = obj.get_state(t)



fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0,:], states[1,:], states[2,:],"-b")

max_range = np.linalg.norm(states[0:3,0])*2

ax.set_xlim(-max_range, max_range)
ax.set_ylim(-max_range, max_range)
ax.set_zlim(-max_range, max_range)
plt.show()

