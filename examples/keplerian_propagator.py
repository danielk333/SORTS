#!/usr/bin/env python

'''
Kepler propagator
==================
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import pyorb
from sorts.common.profiling import Profiler
from sorts.targets.propagator import Kepler

p = Profiler()
p.start('total')

prop = Kepler(profiler=p)

orb = pyorb.Orbit(
    M0=pyorb.M_earth,
    direct_update=True,
    auto_update=True,
    degrees=True,
    a=7000e3, e=0, i=69,
    omega=0, Omega=0, anom=0,
)
print(orb)

t = np.linspace(0, 3600*24.0, num=5000)
mjd0 = Time(53005, format='mjd', scale='utc')

# we can propagate and get ITRS out
prop.out_frame = 'ITRS'
states_itrs = prop.propagate(t, orb, epoch=mjd0)

# or we can set out_frame to GCRS which will cause no 
# transformation to be applied after propagation
prop.out_frame = 'GCRS'
states_gcrs = prop.propagate(t, orb, epoch=mjd0)

# We can also propagate regular cartesian states
states_gcrs_2 = prop.propagate(
    t, 
    orb.cartesian[:, 0], 
    M0=pyorb.M_earth, 
    epoch=mjd0,
)

# or propagate keywords that represent a state
states_gcrs_3 = prop.propagate(
    t, dict(
        a=7000e3, e=0, i=69,
        omega=0, Omega=0, anom=0,
    ), 
    degrees=True,
    M0=pyorb.M_earth,
    epoch=mjd0,
)

p.stop('total')

print(p.fmt(normalize='total'))


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(221, projection='3d')
ax.plot(states_itrs[0, :], states_itrs[1, :], states_itrs[2, :], "-b")
ax.set_title('In: GCRS, out: ITRS, convert: None')

ax = fig.add_subplot(222, projection='3d')
ax.plot(states_gcrs[0, :], states_gcrs[1, :], states_gcrs[2, :], "-b")
ax.set_title('In: GCRS, out: GCRS, convert: None')

ax = fig.add_subplot(223, projection='3d')
ax.plot(states_gcrs_2[0, :], states_gcrs_2[1, :], states_gcrs_2[2, :], "-b")
ax.set_title('In: GCRS-state')

ax = fig.add_subplot(224, projection='3d')
ax.plot(states_gcrs_3[0, :], states_gcrs_3[1, :], states_gcrs_3[2, :], "-b")
ax.set_title('In: GCRS-kepler')

plt.show()
