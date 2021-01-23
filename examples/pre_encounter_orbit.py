#!/usr/bin/env python

'''
Calculating pre-encounter orbits
==================================

'''

import pathlib

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from astropy.time import Time, TimeDelta

import sorts
import pyorb


try:
    pth = pathlib.Path(__file__).parent / 'data' / 'v_ecef.h5'
except NameError:
    import os
    pth = 'data' + os.path.sep + 'v_ecef.h5'

state = np.zeros((6,), dtype=np.float64)
with h5py.File(pth, 'r') as h:
    state[:3] = h['p0'][()]
    state[3:] = h['vel'][()]
    epoch = Time(h['t0'][()], scale='tai', format='unix').utc

print(np.linalg.norm(state[:3])*1e-3)
print(np.linalg.norm(state[3:])*1e-3)
print(epoch.iso)

#This meta kernel lists de430.bsp and de431 kernels as well as others like the leapsecond kernel.
spice_meta = '/home/danielk/IRF/IRF_GITLAB/EPHEMERIS_FILES/MetaK.txt'

states, massive_states, t = sorts.propagate_pre_encounter(
    state, 
    epoch, 
    in_frame = 'ITRS', 
    out_frame = 'HeliocentricMeanEcliptic', 
    termination_check = sorts.distance_termination(dAU = 0.01), #hill sphere of Earth in AU
    spice_meta = spice_meta, 
)

print(f'Time to hill sphere exit: {t[-1]/3600.0:.2f} h')

orb = pyorb.Orbit(
    M0 = pyorb.M_sol,
    direct_update=True,
    auto_update=True,
    degrees = True,
    num = len(t),
)
orb.cartesian = states

kep = orb.kepler

plt.rc('text', usetex=True)

axis_labels = ["$a$ [AU]","$e$ [1]","$i$ [deg]","$\\omega$ [deg]","$\\Omega$ [deg]", "$\\nu$ [deg]" ]
scale = [1/pyorb.AU] + [1]*5

fig = plt.figure(figsize=(15,15))
for i in range(6):
    ax = fig.add_subplot(231+i)
    ax.plot(t/3600.0, kep[i,:]*scale[i], "-b")
    ax.set_xlabel('Time [h]')
    ax.set_ylabel(axis_labels[i])


dt_l = 3600.0

#do a longer propagation to visualize orbit
prop = sorts.propagator.Rebound(
    spice_meta = spice_meta, 
    settings=dict(
        in_frame='HeliocentricMeanEcliptic',
        out_frame='HeliocentricMeanEcliptic',
        time_step = dt_l, #s
        save_massive_states = True, #so we also return all planet positions
    ),
)

fig = plt.figure(figsize=(15,15))
for i in range(6):
    ax = fig.add_subplot(231+i)
    ax.plot(t/3600.0, (states[i,:] - massive_states[i,:,prop.planet_index('Earth')])*1e-3, "-b")


t_l = -np.arange(-3600.0*24, 3600.0*24*365.25*1, dt_l)
states_l, massive_states_l = prop.propagate(
    t_l, 
    states[:,-1], 
    epoch + TimeDelta(t[-1], format='sec'),
    massive_states = massive_states[:,-1,:],
)

orb = pyorb.Orbit(
    M0 = pyorb.M_sol,
    direct_update=True,
    auto_update=True,
    degrees = True,
    num = len(t_l),
)
orb.cartesian = states_l
kep = orb.kepler
fig = plt.figure(figsize=(15,15))
for i in range(6):
    ax = fig.add_subplot(231+i)
    ax.plot(t_l/3600.0, kep[i,:]*scale[i], "-b")
    ax.set_xlabel('Time [h]')
    ax.set_ylabel(axis_labels[i])


fig = plt.figure(figsize=(15,15))
axes = []
for i in range(6):
    axes += [fig.add_subplot(231+i)]
for i, key in enumerate(prop.settings['massive_objects']):
    orb = pyorb.Orbit(
        M0 = pyorb.M_sol,
        direct_update=True,
        auto_update=True,
        degrees = True,
        num = len(t_l),
        m = prop.planets_mass[key],
    )
    orb.cartesian = massive_states_l[:,:,i+1]
    kep = orb.kepler
    for i in range(6):
        ax = axes[i]
        ax.plot(t_l/3600.0, kep[i,:]*scale[i], )
        ax.set_xlabel('Time [h]')
        ax.set_ylabel(axis_labels[i])


fig = plt.figure(figsize=(15,15))
for i in range(6):
    ax = fig.add_subplot(231+i)
    ax.plot(t_l/3600.0, states_l[i,:], "-b")



fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0,:], states[1,:], states[2,:], "-r")
for ind in range(massive_states.shape[2]):
    ax.plot(massive_states[0,:,ind], massive_states[1,:,ind], massive_states[2,:,ind], "-g")

ax.plot(states_l[0,:], states_l[1,:], states_l[2,:], "-b")
for ind in range(massive_states_l.shape[2]):
    ax.plot(massive_states_l[0,:,ind], massive_states_l[1,:,ind], massive_states_l[2,:,ind], "-g")

sorts.plotting.set_axes_equal(ax)

plt.show()