#!/usr/bin/env python

'''
Correlating data with TLE catalog
===================================
'''
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.time import Time

import sorts

radar = sorts.radars.eiscat_uhf

try:
    tle_pth = pathlib.Path(__file__).parent / 'data' / 'uhf_correlation' / 'tle-201801.txt'
    obs_pth = pathlib.Path(__file__).parent / 'data' / 'uhf_correlation' / 'det-000000.h5'
except NameError:
    import os
    tle_pth = 'data' + os.path.sep + 'uhf_correlation' + os.path.sep + 'tle-201801.txt'
    obs_pth = 'data' + os.path.sep + 'uhf_correlation' + os.path.sep + 'det-000000.h5'


# Each entry in the input `measurements` list must be a dictionary that contains the following fields:
#   * 't': [numpy.ndarray] Times relative epoch in seconds
#   * 'r': [numpy.ndarray] Two-way ranges in meters
#   * 'v': [numpy.ndarray] Two-way range-rates in meters per second
#   * 'epoch': [astropy.Time] epoch for measurements
#   * 'tx': [sorts.TX] Pointer to the TX station
#   * 'rx': [sorts.RX] Pointer to the RX station
print('Loading EISCAT UHF ENVISAT Data')
with h5py.File(str(obs_pth),'r') as h_det:
    r = h_det['r'][()]*1e3 #km -> m, one way
    t = h_det['t'][()] #Unix seconds
    v = -h_det['v'][()] #Inverted definition of range rate, one way

    t = Time(t, format='unix', scale='utc')
    epoch = t[0]
    t = (t - epoch).sec

    dat = {
        'r': r*2,
        't': t,
        'v': v*2,
        'epoch': epoch,
        'tx': radar.tx[0],
        'rx': radar.rx[0],
    }

print('Loading TLE population')
pop = sorts.population.tle_catalog(tle_pth)
pop.out_frame = 'ITRS'

pop.filter('oid', lambda oid: np.abs(oid - 43075) < 4)

print('population size: {}'.format(len(pop)))

print('Correlating data and population')
indecies, metric, cdat = sorts.correlate(
    measurements = [dat],
    population = pop,
    n_closest = 2,
)

print('Match metric:')
for ind, dst in zip(indecies, metric):
    print(f'ind = {ind}: metric = {dst}')


def plot_correlation(dat, cdat):
    '''Plot the correlation between the measurement and simulated population object.
    '''
    t = dat['t']
    r = dat['r']
    v = dat['v']
    r_ref = cdat['r_ref']
    v_ref = cdat['v_ref']

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(211)
    ax.plot(t - t[0], r*1e-3, label='measurement')
    ax.plot(t - t[0], r_ref*1e-3, label='simulation')
    ax.set_ylabel('Range [km]')
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(212)
    ax.plot(t - t[0], v*1e-3, label='measurement')
    ax.plot(t - t[0], v_ref*1e-3, label='simulation')
    ax.set_ylabel('Velocity [km/s]')
    ax.set_xlabel('Time [s]')

    plt.legend()

    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(221)
    ax.hist((r_ref - r)*1e-3)
    ax.set_xlabel('Range residuals [km]')

    ax = fig.add_subplot(222)
    ax.hist((v_ref - v)*1e-3)
    ax.set_xlabel('Velocity residuals [km/s]')
    
    ax = fig.add_subplot(223)
    ax.plot(t - t[0], (r_ref - r)*1e-3)
    ax.set_ylabel('Range residuals [km]')
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(224)
    ax.plot(t - t[0], (v_ref - v)*1e-3)
    ax.set_ylabel('Velocity residuals [km/s]')
    ax.set_xlabel('Time [s]')

plot_correlation(dat, cdat[0][0])
plot_correlation(dat, cdat[1][0])

plt.show()
