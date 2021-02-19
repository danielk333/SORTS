#!/usr/bin/env python

'''
Calculating distributions pre-encounter orbits
=================================================

'''

import pathlib

import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from tqdm import tqdm

from astropy.time import Time, TimeDelta

import sorts
import pyorb

num = 100

try:
    pth = pathlib.Path(__file__).parent / 'data' / 'v_ecef.h5'
except NameError:
    import os
    pth = 'data' + os.path.sep + 'v_ecef.h5'

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

def sample_orbit(state):
    states, massive_states, t = sorts.propagate_pre_encounter(
        state, 
        epoch, 
        in_frame = 'ITRS', 
        out_frame = 'HCRS', 
        termination_check = sorts.distance_termination(dAU = 0.01), #hill sphere of Earth in AU
        kernel = kernel, 
    )

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

    return t, orb.kepler.copy()

samples = []
for i in tqdm(range(num)):
    dstate = state.copy()
    dstate[3:] += np.random.randn(3)*vel_stds

    t, kep = sample_orbit(dstate)
    samples.append((t,kep))


plt.rc('text', usetex=True)

axis_labels = ["$a$ [AU]","$e$ [1]","$i$ [deg]","$\\omega$ [deg]","$\\Omega$ [deg]", "$\\nu$ [deg]" ]
scale = [1/pyorb.AU] + [1]*5

fig = plt.figure(figsize=(15,15))
for i in range(6):
    ax = fig.add_subplot(231+i)
    for j in range(num):
        t, kep = samples[j]
        ax.plot(t/3600.0, kep[i,:]*scale[i], "-b", alpha=0.1)
    ax.set_xlabel('Time [h]')
    ax.set_ylabel(axis_labels[i])
fig.suptitle('Propagation to pre-encounter elements')

plt.show()