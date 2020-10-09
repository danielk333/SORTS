#!/usr/bin/env python

'''
Loading a TLE catalog
=============================================
'''
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import sorts
from sorts import plotting
from sorts.population import tle_catalog

try:
    pth = pathlib.Path(__file__).parent / 'data' / 'tle.txt'
except NameError:
    import os
    pth = 'data' + os.path.sep + 'tle.txt'

pop = tle_catalog(pth, kepler=True)

print(pop.print(n=slice(None,10), fields = ['oid','a','e','i','mjd0', 'm', 'd', 'BSTAR']))

plotting.orbits(
    pop.get_fields(['x','y','z','vx','vy','vz'], named=False),
    title =  "State distribution of tle catalog",
    axis_labels = 'earth-state',
    limits = [(-3,3)]*3 + [(-15,15)]*3,
)

plotting.orbits(
    pop.get_fields(['a','e','i','aop','raan','mu0'], named=False),
    title =  "Orbit distribution of tle catalog",
    axis_labels = 'earth-orbit',
    limits = [(0, 5)] + [(None, None)]*5,
)

#look at kepler elements
orbit = pop.get_orbit(n=101, fields = ['x','y','z','vx','vy','vz', 'm'])
print(f'\n Orbit of satnum: {pop["oid"][101]} \n{str(orbit)}\n')

#we can also create a space object
obj = pop.get_object(n=101)
print(obj)

#lets get ITRS states
obj.propagator.set(out_frame='ITRS')

t = np.linspace(0,3600*24.0*2,num=5000)
states = obj.get_state(t)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0,:], states[1,:], states[2,:],"-b")

plt.show()
