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
    pth = pathlib.Path('.').parent / 'data' / 'v_ecef.h5'


state = np.zeros((6,), dtype=np.float64)
with h5py.File(pth, 'r') as h:
    state[:3] = h['p0'][()]
    state[3:] = h['vel'][()]
    vel_stds = h['vel_sigma'][()]
    epoch = Time(h['t0'][()], scale='tai', format='unix').utc

print(f'Encounter Geocentric range (ITRS): {np.linalg.norm(state[:3])*1e-3} km')
print(f'Encounter Geocentric speed (ITRS): {np.linalg.norm(state[3:])*1e-3} km/s')
print(f'At epoch: {epoch.iso}')

kernel = '/home/danielk/IRF/IRF_GITLAB/EPHEMERIS_FILES/de430.bsp'

print(f'Using JPL kernel: {kernel}')

states, massive_states, t = sorts.propagate_pre_encounter(
    state, 
    epoch, 
    in_frame = 'ITRS', 
    out_frame = 'HCRS', 
    termination_check = sorts.distance_termination(dAU = 0.01), #hill sphere of Earth in AU
    kernel = kernel, 
)

print(f'Time to hill sphere exit: {t[-1]/3600.0:.2f} h')

states_HMC = sorts.frames.convert(
    epoch + TimeDelta(t, format='sec'),
    states, 
    in_frame = 'HCRS', 
    out_frame = 'HeliocentricMeanEcliptic',
)

orb = pyorb.Orbit(
    M0 = pyorb.M_sol,
    direct_update=True,
    auto_update=True,
    degrees = True,
    num = len(t),
)
orb.cartesian = states_HMC

kep = orb.kepler

print('Pre-encounter orbit:')
print(orb[-1])

plt.rc('text', usetex=True)

axis_labels = ["$a$ [AU]","$e$ [1]","$i$ [deg]","$\\omega$ [deg]","$\\Omega$ [deg]", "$\\nu$ [deg]" ]
scale = [1/pyorb.AU] + [1]*5

fig = plt.figure(figsize=(15,15))
for i in range(6):
    ax = fig.add_subplot(231+i)
    ax.plot(t/3600.0, kep[i,:]*scale[i], "-b")
    ax.set_xlabel('Time [h]')
    ax.set_ylabel(axis_labels[i])
fig.suptitle('Propagation to pre-encounter elements')

dt_l = 3600.0*12

#do a longer propagation to visualize orbit
prop = sorts.propagator.Rebound(
    kernel = kernel, 
    settings=dict(
        in_frame='HCRS',
        out_frame='HCRS',
        time_step = dt_l, #s
        save_massive_states = True, #so we also return all planet positions
    ),
)

t_l = -np.arange(0, 3600.0*24*365.25*20, dt_l)
states_l, massive_states_l = prop.propagate(
    t_l, 
    states[:,-1], 
    epoch + TimeDelta(t[-1], format='sec'),
    massive_states = massive_states[:,-1,:],
)

states_l_HMC = sorts.frames.convert(
    epoch + TimeDelta(t_l, format='sec') + TimeDelta(t[-1], format='sec'),
    states_l, 
    in_frame = 'HCRS', 
    out_frame = 'HeliocentricMeanEcliptic',
)

massive_states_l_HMC = massive_states_l.copy()
for i in range(massive_states_l_HMC.shape[2]):
    massive_states_l_HMC[:,:,i] = sorts.frames.convert(
        epoch + TimeDelta(t_l, format='sec') + TimeDelta(t[-1], format='sec'),
        massive_states_l[:,:,i], 
        in_frame = 'HCRS', 
        out_frame = 'HeliocentricMeanEcliptic',
    )

orb = pyorb.Orbit(
    M0 = pyorb.M_sol,
    direct_update=True,
    auto_update=True,
    degrees = True,
    num = len(t_l),
)
orb.cartesian = states_l_HMC
kep = orb.kepler
fig = plt.figure(figsize=(15,15))
for i in range(6):
    ax = fig.add_subplot(231+i)
    ax.plot(t_l/(3600.0*24*365.25), kep[i,:]*scale[i], "-b")
    ax.set_xlabel('Time [y]')
    ax.set_ylabel(axis_labels[i])
fig.suptitle('Long term backwards elements')

fig = plt.figure(figsize=(15,15))
axes = []
for i in range(6):
    axes += [fig.add_subplot(231+i)]
for i, key in enumerate(prop.settings['massive_objects']):
    if key == 'Sun':
        continue
    orb = pyorb.Orbit(
        M0 = pyorb.M_sol,
        direct_update=True,
        auto_update=True,
        degrees = True,
        num = len(t_l),
        m = prop.planets_mass[key],
    )
    orb.cartesian = massive_states_l_HMC[:,:,i]
    kep = orb.kepler
    for i in range(6):
        ax = axes[i]
        ax.plot(t_l/(3600.0*24*365.25), kep[i,:]*scale[i], label=key)
        ax.set_xlabel('Time [y]')
        ax.set_ylabel(axis_labels[i])

fig.suptitle('Solarsystem elements')

axes[-1].legend()


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

ax.plot(states_l_HMC[0,:], states_l_HMC[1,:], states_l_HMC[2,:], "-b")
for ind in range(massive_states_l_HMC.shape[2]):
    ax.plot(massive_states_l_HMC[0,:,ind], massive_states_l_HMC[1,:,ind], massive_states_l_HMC[2,:,ind], "-g")

sorts.plotting.set_axes_equal(ax)

plt.show()