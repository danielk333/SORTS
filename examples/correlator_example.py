#!/usr/bin/env python

'''
Correlating data with TLE catalog
===================================

TO RUN THIS EXAMPLE, YOU NEED THE CELESTRACK CATALOG FROM 2018-01-01 in the 'data/uhf_correlation/tle-201801.txt' path
'''
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.time import Time

import sorts

radar = sorts.radars.eiscat_uhf

#TO RUN THIS EXAMPLE, YOU NEED THE CELESTRACK CATALOG FROM 2018-01-01


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
print('Loading EISCAT UHF monostatic measurements')
with h5py.File(str(obs_pth),'r') as h_det:
    r = h_det['r'][()]*1e3 #km -> m, one way
    t = h_det['t'][()] #Unix seconds
    v = -h_det['v'][()] #Inverted definition of range rate, one way

    inds = np.argsort(t)
    t = t[inds]
    r = r[inds]
    v = v[inds]

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
pop = sorts.population.tle_catalog(tle_pth, cartesian=False)

#correlate requires output in ECEF 
pop.out_frame = 'ITRS'

#filter out some objects for speed (leave 100 random ones)
#IRIDIUM = 43075U
random_selection = np.random.randint(low=0, high=len(pop), size=(100,))
random_selection = pop['oid'][random_selection].tolist()

pop.filter('oid', lambda oid: oid == 43075 or oid in random_selection)

#Lets also remove all but 1 TLE for IRIDIUM
#Using some numpy magic filtering
keep = np.full((len(pop),), True, dtype=np.bool)
mjds = pop.data[pop.data['oid'] == 43075]['mjd0']
best_mjd = np.argmin(np.abs(mjds - epoch.mjd))
best_mjd = mjds[best_mjd]
keep[pop.data['oid'] == 43075] = False
keep[np.logical_and(pop.data['oid'] == 43075, pop.data['mjd0'] == best_mjd)] = True
pop.data = pop.data[keep]


print('population size: {}'.format(len(pop)))

print('Correlating data and population')
indecies, metric, cdat = sorts.correlate(
    measurements = [dat],
    population = pop,
    n_closest = 4,
)

print('Match metric:')
for ind, dst in zip(indecies, metric):
    print(f'ind = {ind}: metric = {dst}')
    print(pop.print(n=ind, fields=['oid', 'mjd0', 'line1', 'line2']) + '\n')


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
plot_correlation(dat, cdat[3][0])

plt.show()
