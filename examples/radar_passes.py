#!/usr/bin/env python

'''

'''

import numpy as np
import matplotlib.pyplot as plt
import pyorb

import sorts
from sorts.propagator import Orekit
from sorts.radar.instances import eiscat3d


orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees=True, a=7200e3, e=0.1, i=75, omega=0, Omega=79, anom=72, epoch=53005.0)
print(orb)

t = sorts.equidistant_sampling(
    orbit = orb, 
    start_t = 0, 
    end_t = 3600*24*1, 
    max_dpos=1e3,
)

orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'
prop = Orekit(
    orekit_data = orekit_data, 
    settings=dict(
        in_frame='EME',
        out_frame='ITRF',
        drag_force = False,
        radiation_pressure = False,
        max_step=10.0,
    ),
)
print(f'Temporal points: {len(t)}')
states = prop.propagate(t, orb.cartesian[:,0], orb.epoch, A=1.0, C_R = 1.0, C_D = 1.0)

passes = eiscat3d.find_passes(t, states)

fig = plt.figure(figsize=(15,15))
axes = [
    [
        fig.add_subplot(221, projection='3d'),
        fig.add_subplot(222),
    ],
    [
        fig.add_subplot(223),
        fig.add_subplot(224),
    ],
]

for pi, ps in enumerate(passes[0][0]):
    zang = ps.zenith_angle()
    axes[0][0].plot(ps.enu[0][0,:], ps.enu[0][1,:], ps.enu[0][2,:], '-', label=f'pass-{pi}')
    axes[0][1].plot((ps.t - ps.start())/3600.0, zang[0], '-', label=f'pass-{pi}')
    axes[1][0].plot((ps.t - ps.start())/3600.0, ps.range()[0], '-', label=f'pass-{pi}')
    axes[1][1].plot((ps.t - ps.start())/3600.0, zang[1], '-', label=f'pass-{pi}')

axes[1][1].legend()
plt.show()
