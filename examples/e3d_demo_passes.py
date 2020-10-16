#!/usr/bin/env python

'''
Finding passes over EISCAT 3D Demo
====================================

'''

import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pyorb
from astropy.time import Time, TimeDelta

import sorts
from sorts.population import tle_catalog

eiscat3d = sorts.radars.eiscat3d_demonstrator_interp

print(f'lat={eiscat3d.tx[0].lat:.2f} deg, lon={eiscat3d.tx[0].lon:.2f} deg')

#############
# CHOOSE OBJECTS
#############

profiler = sorts.profiling.Profiler()
logger = sorts.profiling.get_logger()

# ENVISAT
tles = [
    (
        '1 27386U 02009A   20290.00854375  .00000022  00000-0  20588-4 0  9993',
        '2 27386  98.1398 290.7332 0001195  90.6832 325.5676 14.37995280975942',
     ),
]

pop = tle_catalog(tles, kepler=False)

pop.propagator_options['settings']['out_frame'] = 'ITRS' #output states in ECEF

obj = pop.get_object(0)
obj.parameters['d'] = 2.0

epoch = obj.epoch
print(epoch.iso)
print(obj)

t = np.arange(0, 3600.0*24.0*10, 10.0)

print(f'Temporal points: {len(t)}')
states = obj.get_state(t)

passes = eiscat3d.find_passes(t, states)

for pi, ps in enumerate(passes[0][0]): #tx-0 and rx-0
    date_ = (epoch + TimeDelta(ps.start(), format='sec')).iso
    date_end_ = (epoch + TimeDelta(ps.end(), format='sec')).iso
    zang = ps.zenith_angle()
    print(f'Pass id={pi} (min-zang={np.min(zang):.1f} deg) -> ' + str(date_) + ' UTC -> ' + str(date_end_) + ' UTC')


select_passes = [22,29]

ax = sorts.plotting.local_passes([passes[0][0][pi] for pi in select_passes])


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states[0,:], states[1,:], states[2,:])

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
    if pi not in select_passes:
        continue
    zang = ps.zenith_angle()
    snr = ps.calculate_snr(eiscat3d.tx[0], eiscat3d.rx[0], diameter=obj.d)

    axes[0][0].plot(ps.enu[0][0,:], ps.enu[0][1,:], ps.enu[0][2,:], '-', label=f'pass-{pi}')
    axes[0][0].set_xlabel('East-North-Up coordinates')

    axes[0][1].plot((ps.t - ps.start())/60.0, zang[0], '-', label=f'pass-{pi}')
    axes[0][1].set_xlabel('Time past rise time [min]')
    axes[0][1].set_ylabel('Zenith angle from TX [deg]')

    axes[1][0].plot((ps.t - ps.start())/60.0, ps.range()[0]*1e-3, '-', label=f'pass-{pi}')
    axes[1][0].set_xlabel('Time past rise time [min]')
    axes[1][0].set_ylabel('Range from TX [km]')
    
    ax2 = axes[1][0].twinx()  # instantiate a second axes that shares the same x-axis

    ax2.set_ylabel('Range rate from TX [km/s]')  # we already handled the x-label with ax1
    ax2.plot((ps.t - ps.start())/60.0, ps.range_rate()[0]*1e-3, '-')
    ax2.tick_params(axis='y')

    axes[1][1].plot((ps.t - ps.start())/60.0, 10*np.log10(snr), '-', label=f'pass-{pi}')
    axes[1][1].set_xlabel('Time past rise time [min]')
    axes[1][1].set_ylabel('Optimal SNR/0.1s [dB]')

axes[1][1].legend()
plt.show()
