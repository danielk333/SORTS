#!/usr/bin/env python

'''
REBOUND propagator usage
================================
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta

from sorts.propagator import Rebound
import pyorb

#This meta kernel lists de430.bsp and de431 kernels as well as others like the leapsecond kernel.
spice_meta = '/home/danielk/IRF/IRF_GITLAB/EPHEMERIS_FILES/MetaK.txt'

#We input in International Terrestrial Reference System coordinates
#and output in International Celestial Reference System
#going trough HeliocentricMeanEclipticJ2000 internally in Rebound
prop = Rebound(
    spice_meta = spice_meta, 
    settings=dict(
        in_frame='ITRS',
        out_frame='ICRS',
        time_step = 30.0, #s
        use_sim_geocentric = True,
    ),
)

n_test = 10

print(prop)

orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees = True, a=8000e3, e=0, i=69, omega=0, Omega=0, anom=0)
print('Initial orbit:')
print(orb)

state0 = np.squeeze(orb.cartesian)
t = np.linspace(0,3600*12,num=300)
epoch0 = Time(53005, format='mjd', scale='utc')
times = epoch0 + TimeDelta(t, format='sec')

states = prop.propagate(t, state0, epoch0)

#the propagator remembers the latest propagation
print(prop)

#for reproducibility
np.random.seed(293489776)

#we can also propagate several test partcies at the same time
state0_r = np.zeros((6, n_test))
state0_r = state0_r + state0[:, None]
state0_r[3:,:] += np.random.randn(3,n_test)*1e3 #1km/s std separation

states_r = prop.propagate(t, state0_r, epoch0)


#plot results
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(121, projection='3d')
ax.plot(states[0,:], states[1,:], states[2,:],"-r", label='Base test particle')
#the earth states are also available in prop.earth_states if they were enabled in the settings
#These are also converted to the output frame if it is not geocentric so that one easily

ax.plot(prop.earth_states[0,:], prop.earth_states[1,:], prop.earth_states[2,:],".g", label='Earth')
ax.plot(states_r[0,:,0], states_r[1,:,0], states_r[2,:,0],"-b", label='Perturbed test particle')
for i in range(1,n_test):
    ax.plot(states_r[0,:,i], states_r[1,:,i], states_r[2,:,i],"-b")

ax.legend()

#lets also plot it in geocentric for reference
#by converting manually
states_geo = states.copy() - prop.earth_states
states_r_geo = states_r.copy()  - prop.earth_states[:,:,None]

ax = fig.add_subplot(122, projection='3d')
ax.plot(states_geo[0,:], states_geo[1,:], states_geo[2,:],"-r")
ax.plot([0], [0], [0],".g")
for i in range(n_test):
    ax.plot(states_r_geo[0,:,i], states_r_geo[1,:,i], states_r_geo[2,:,i],"-b")


plt.show()