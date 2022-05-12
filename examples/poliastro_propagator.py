#!/usr/bin/env python

'''
Poliastro TwoBody propagator
=============================
'''

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import pyorb
from sorts.profiling import Profiler
from sorts.propagator import TwoBody
from sorts.propagator import Kepler

p = Profiler()
p.start('total')

prop = TwoBody(profiler=p)
prop_kep = Kepler(profiler=p)

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
epoch = Time(53005, format='mjd', scale='utc')
state0 = orb.cartesian[:, 0]

# we can propagate and get ITRS out
prop.out_frame = 'ITRS'
prop_kep.out_frame = 'ITRS'
states_itrs = prop.propagate(t, state0, epoch=epoch)
states_itrs_kep = prop_kep.propagate(t, orb, epoch=epoch)

# or we can set out_frame to GCRS which will cause no 
# transformation to be applied after propagation
prop.out_frame = 'GCRS'
prop_kep.out_frame = 'GCRS'
states_gcrs = prop.propagate(t, state0, epoch=epoch)
states_gcrs_kep = prop_kep.propagate(t, orb, epoch=epoch)

p.stop('total')
print(p.fmt(normalize='total'))


fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(221, projection='3d')
ax.plot(states_itrs[0, :], states_itrs[1, :], states_itrs[2, :], "-b")
ax.set_title('Poliastro Frame: ITRS')

ax = fig.add_subplot(222, projection='3d')
ax.plot(states_gcrs[0, :], states_gcrs[1, :], states_gcrs[2, :], "-b")
ax.set_title('Poliastro Frame: GCRS')

ax = fig.add_subplot(223, projection='3d')
ax.plot(states_itrs_kep[0, :], states_itrs_kep[1, :], states_itrs_kep[2, :], "-b")
ax.set_title('Pyorb Frame: ITRS')

ax = fig.add_subplot(224, projection='3d')
ax.plot(states_gcrs_kep[0, :], states_gcrs_kep[1, :], states_gcrs_kep[2, :], "-b")
ax.set_title('Pyorb Frame: GCRS')

plt.show()
