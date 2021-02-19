#!/usr/bin/env python

'''
REBOUND propagator usage
================================
'''
import pathlib
import configparser

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta

from sorts.propagator import Rebound
import pyorb


try:
    base_pth = pathlib.Path(__file__).parents[1].resolve()
except NameError:
    base_pth = pathlib.Path('.').parents[1].resolve()

config = configparser.ConfigParser(interpolation=None)
config.read([base_pth / 'example_config.conf'])
kernel = pathlib.Path(config.get('rebound_propagator.py', 'kernel'))

if not kernel.is_absolute():
    kernel = base_pth / kernel.relative_to('.')


#We input in International Terrestrial Reference System coordinates
#and output in International Celestial Reference System
#going trough HeliocentricMeanEclipticJ2000 internally in Rebound
prop = Rebound(
    kernel = kernel, 
    settings=dict(
        in_frame='ITRS',
        out_frame='ICRS',
        time_step = 30.0, #s
        save_massive_states = True, #so we also return all planet positions
    ),
)

n_test = 10

print(prop)

orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees = True, a=8000e3, e=0, i=69, omega=0, Omega=0, anom=0)
print('Initial orbit:')
print(orb)

state0 = np.squeeze(orb.cartesian)
t = np.linspace(-3600*6,3600*6,num=300) #we can do both forward and backward
epoch0 = Time(53005, format='mjd', scale='utc')
times = epoch0 + TimeDelta(t, format='sec')

states, massive_states = prop.propagate(t, state0, epoch0)

#the propagator remembers the latest propagation
print(prop)

#for reproducibility
np.random.seed(293489776)

#we can also propagate several test partcies at the same time
state0_r = np.zeros((6, n_test))
state0_r = state0_r + state0[:, None]
state0_r[3:,:] += np.random.randn(3,n_test)*0.5e3 #1km/s std separation

#this is inefficient since we could have included the first particle
#but for the exemplification sake we propagate again
states_r, _ = prop.propagate(t, state0_r, epoch0)


#plot results
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(121, projection='3d')
ax.plot(states[0,:], states[1,:], states[2,:],"-r", label='Base test particle')


#we get the index of the earth using the convenience function
E_ind = prop.planet_index('Earth')

ax.plot(massive_states[0,:,E_ind], massive_states[1,:,E_ind], massive_states[2,:,E_ind],".g", label='Earth')
ax.plot(states_r[0,:,0], states_r[1,:,0], states_r[2,:,0],"-b", label='Perturbed test particle')
for i in range(1,n_test):
    ax.plot(states_r[0,:,i], states_r[1,:,i], states_r[2,:,i],"-b")

ax.legend()

#lets also plot it in geocentric for reference
#by converting manually
earth_states = massive_states[:,:,E_ind]

states_geo = states.copy() - earth_states
states_r_geo = states_r.copy() - earth_states[:,:,None]

ax = fig.add_subplot(122, projection='3d')
ax.plot(states_geo[0,:], states_geo[1,:], states_geo[2,:],"-r")
ax.plot([0], [0], [0],".g")
for i in range(n_test):
    ax.plot(states_r_geo[0,:,i], states_r_geo[1,:,i], states_r_geo[2,:,i],"-b")

plt.show()