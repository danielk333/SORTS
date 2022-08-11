#!/usr/bin/env python

'''
A controller for tracing
======================================

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyorb


import sorts
eiscat3d = sorts.radars.eiscat3d
from sorts.radar.controllers import Tracker
from sorts.targets.propagator import SGP4
from sorts.common.profiling import Profiler

p = Profiler()
p.start('total')

prop = SGP4(
    settings = dict(
        out_frame='ITRF',
    ),
)

orb = pyorb.Orbit(M0=pyorb.M_earth, direct_update=True, auto_update=True, degrees=True, a=6700e3, e=0, i=75, omega=0, Omega=80, anom=72)
t = np.linspace(0,132,num=11)
mjd0 = 53005

p.start('propagate')
states = prop.propagate(t, orb.cartesian[:,0], mjd0, A=1.0, C_R = 1.0, C_D = 1.0)
p.stop('propagate')

e3d = Tracker(profiler=p)
controls = e3d.generate_controls(t, eiscat3d, t, states[:3,:], t_slice=10.0)

for period_id in range(controls.n_periods):
    sorts.plotting.plot_control_sequence([controls], 0, 120, 60)

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0,:-1], states[1,:-1], states[2,:-1],"or")

for tx in controls.radar.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in controls.radar.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

for period_id in range(controls.n_periods):
    p.start('Tracker-plot')
    sorts.plotting.plot_beam_directions(controls.get_pdirs(period_id), eiscat3d, ax=ax, profiler=p, tx_beam=True, rx_beam=True, zoom_level=0.9, azimuth=10, elevation=10)
    p.stop('Tracker-plot')

sorts.plotting.grid_earth(ax)

p.stop('total')
print(p.fmt(normalize='total'))

plt.show()