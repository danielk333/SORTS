#!/usr/bin/env python

'''
Correlating data with TLE catalog
===================================

TO RUN THIS EXAMPLE, YOU NEED THE CELESTRACK CATALOG FROM 2018-01-01 CONFIGURED IN "example_config.conf"
'''
import configparser
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import h5py
from astropy.time import Time

import sorts

radar = sorts.radars.eiscat_uhf

try:
    base_pth = pathlib.Path(__file__).parents[1].resolve()
except NameError:
    base_pth = pathlib.Path('.').parents[1].resolve()

config = configparser.ConfigParser(interpolation=None)
config.read([base_pth / 'example_config.conf'])
tle_pth = pathlib.Path(config.get('correlator_example.py', 'tle_catalog'))
obs_pth = pathlib.Path(config.get('correlator_example.py', 'observation_data'))

if not tle_pth.is_absolute():
    tle_pth = base_pth / tle_pth.relative_to('.')
if not obs_pth.is_absolute():
    obs_pth = base_pth / obs_pth.relative_to('.')


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
        'r': r[10:]*2,
        't': t[10:],
        'v': v[10:]*2,
        'epoch': epoch,
        'tx': radar.tx[0],
        'rx': radar.rx[0],
    }
    dat2 = {
        'r': r[0:10]*2,
        't': t[0:10],
        'v': v[0:10]*2,
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
    measurements = [dat,dat2],
    population = pop,
    n_closest = 4,
)

print('Match metric:')
for ind, dst in zip(indecies, metric):
    print(f'ind = {ind}: metric = {dst}')
    print(pop.print(n=ind, fields=['oid', 'mjd0', 'line1', 'line2']) + '\n')


#
# Lets try correlating on each individual measurement instead
#


def vector_diff_metric(t, r, v, r_ref, v_ref):
    '''Return a vector of absolute differences
    '''
    return np.abs(r_ref - r) + np.abs(v_ref - v)

indecies0, metric0, cdat0 = sorts.correlate(
    measurements = [dat2],
    population = pop,
    n_closest = 3,
    metric=vector_diff_metric, 
    metric_reduce=None, #since we don't reduce, it assumes we are doing correlation measurement-wise
)

print('Individual measurement match metric:')
for mind, (ind, dst) in enumerate(zip(indecies0.T, metric0.T)):
    print(f'measurement = {mind} | object ind = {ind} | metric = {dst}')


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
