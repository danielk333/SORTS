#!/usr/bin/env python

'''
============
Using Frames
============

This example showcases the use of the ``sorts.frames`` module for coordinate 
frame transformations.
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pyorb

from astropy.utils import iers
from astropy.time import Time
iers.conf.auto_download = False

from sorts.targets.propagator import SGP4
from sorts.transformations import frames

# propagator setup
prop = SGP4()
orb = pyorb.Orbit(
    M0 = pyorb.M_earth,
    direct_update=True,
    auto_update=True,
    degrees = True,
    a=7000e3,
    e=0,
    i=69,
    omega=0,
    Omega=0,
    anom=0,
)

# propagation time array
t = np.linspace(0,3600*24.0,num=5000)
mjd0 = Time(53005.0, format='mjd', scale='utc')
times = Time(mjd0 + t/(3600*24.0), format='mjd', scale='utc')

# propagate states in the TEME reference frame
states_teme = prop.propagate(t, orb.cartesian[:,0], epoch=mjd0, A=1.0, C_R = 1.0, C_D = 1.0)

# plot states in the 'TEME', 'ITRS', 'ICRS' and 'GCRS' reference frames
fig = plt.figure(figsize=(15,15))
for ind, frame_name in enumerate(['TEME', 'ITRS', 'ICRS', 'GCRS']):
    # conversion between reference frames
    states_conv = frames.convert(
        times, 
        states_teme, 
        in_frame='TEME', 
        out_frame=frame_name,
    )

    # plot
    ax = fig.add_subplot(221 + ind, projection='3d')
    ax.plot(states_conv[0,:], states_conv[1,:], states_conv[2,:],"-b")
    ax.set_title(f'In: TEME, out: {frame_name}')

plt.show()