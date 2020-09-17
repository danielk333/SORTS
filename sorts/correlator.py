#!/usr/bin/env python

'''Correlate measurement time series with a population of objects to find the best match.

Currently only works for Mono-static measurements.

# TODO: Assume a uniform prior distribution over population index, posterior distribution is the probability of what object generated the data. Probability comes from measurement covariance.
'''

import sys
import os
import time
import glob

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

import numpy as np
import scipy
import h5py
import scipy.constants



def residual_distribution_metric(t, r, v, r_ref, v_ref):
    '''Using the simulated and the measured ranges and rage-rates calculate a de-correlation metric.
    
    :param numpy.ndarray t: Times in seconds corresponding to measurement and simulation data.
    :param numpy.ndarray r: Measured ranges in meters
    :param numpy.ndarray v: Measured rage-rates in meters per second
    :param numpy.ndarray r_ref: Simulated ranges in meters
    :param numpy.ndarray v_ref: Simulated rage-rates in meters per second
    :return: Metric value, smaller values indicate better match.
    :rtype: float
    '''
    residual_r_mu = np.mean(r_ref - r)
    residual_v_mu = np.mean(v_ref - v)
    
    metric = np.abs(residual_r_mu) + np.abs(residual_v_mu)

    return metric


def correlate(data, station, population, metric, n_closest = 1, out_file = None, verbose=False, MPI_on=False):
    '''Given a mono-static measurement of ranges and rage-rates, a radar model and a population: correlate measurements with population.

    :param dict data: Dictionary that contains measurement data. Contents are described below.
    :param AntennaRX station: Model of receiver station that performed the measurement.
    :param Population population: Population to correlate against.
    :param function metric: Metric used to correlate measurement and simulation of population.
    :param int n_closest: Number of closest matches to output.
    :param str out_file: If not :code:`None`, save the output data to this path.
    :param bool MPI_on: If True use internal parallelization with MPI to calculate correlation. Turn to False to externally parallelize with MPI.
    

    **Measurement data:**

      The file must be a dictionary that contains three data-sets:
        * 't': Times in unix-seconds
        * 'r': Ranges in meters
        * 'v': Range-rates in meters per second
      They should all be numpy vectors of equal length.

    '''
    raise NotImplementedError('TODO')
    
    r = data['r']
    t = data['t']
    v = data['v']

    t_sort = t.argsort()
    t = t[t_sort]
    r = r[t_sort]
    v = v[t_sort]
    
    #lt correction
    lt_correction = r/scipy.constants.c
    t += lt_correction

    _day = 3600.0*24.0

    loc_ecef = station.ecef.copy()
    loc_ecef_norm = loc_ecef/np.linalg.norm(loc_ecef)

    jd_check = dpt.unix_to_jd(t)

    r_ref = np.empty(r.shape, dtype=r.dtype)
    v_ref = np.empty(v.shape, dtype=v.dtype)

    correlation_data = {}
    SO_generator = population.object_generator()

    if MPI_on:
        step = MPI.COMM_WORLD.size
        next_check = MPI.COMM_WORLD.rank
    else:
        step = 1
        next_check = 0

    for ind, obj in enumerate(SO_generator):
        if ind == next_check:
            if verbose:
                print('\n\nPID {} correlating {}/{} -------'.format(MPI.COMM_WORLD.rank, ind+1, len(population)))
                print(obj)
            
            jd_obj = dpt.mjd_to_jd(obj.mjd0)
            
            t_check = (jd_check - jd_obj)*_day
            states = obj.get_state(t_check)

            for jdi in range(jd_check.size):
                r_tmp = loc_ecef - states[:3,jdi]

                r_ref[jdi] = np.linalg.norm(r_tmp)
                v_ref[jdi] = np.dot(
                    states[3:,jdi],
                    r_tmp/r_ref[jdi],
                )

            residual_r_mu = np.mean(r_ref - r)
            residual_r_std = np.std(r_ref - r)
            residual_v_mu = np.mean(v_ref - v)
            residual_v_std = np.std(v_ref - v)
            if len(v_ref) > 1:
                residual_a = ((v_ref[-1] - v_ref[0]) - v[-1] - v[0])/(t[-1] - t[0])
            else:
                residual_a = 0

            if verbose:
                print('Residuals:')
                print('residual_r_mu  = {} m'.format(residual_r_mu))
                print('residual_r_std = {} m'.format(residual_r_std))
                print('residual_v_mu  = {} m/s'.format(residual_v_mu))
                print('residual_v_std = {} m/s'.format(residual_v_std))
                print('residual_a     = {} m/s^2'.format(residual_a))
            
            cdat = {
                    'r_ref': r_ref.copy(),
                    'v_ref': v_ref.copy(),
                    'sat_id': obj.oid,
                    'stat': [residual_r_mu, residual_r_std, residual_v_mu, residual_v_std, residual_a]
                }
            if obj.oid in correlation_data:
                correlation_data[obj.oid].append(cdat)
            else:
                correlation_data[obj.oid] = [cdat]

            next_check += step

    oids = population['oid']

    if step > 1:

        if MPI.COMM_WORLD.rank == 0:
            if verbose:
                print('---> PID %i: Receiving all results <barrier>'%(MPI.COMM_WORLD.rank))

            for T in range(1,MPI.COMM_WORLD.size):
                for ID in range(T,len(population),MPI.COMM_WORLD.size):
                    oid = oids[ID]
                    correlation_data[oid] = MPI.COMM_WORLD.recv(source=T, tag=oid)
                    if verbose:
                        print('PID{} recived packet {} from PID{} '.format(MPI.COMM_WORLD.rank, oid, T))
        else:
            if verbose:
                print('---> PID %i: Distributing all correlation results to process 0 <barrier>'%(MPI.COMM_WORLD.rank))

            for ID in range(MPI.COMM_WORLD.rank,len(population),MPI.COMM_WORLD.size):
                oid = oids[ID]
                MPI.COMM_WORLD.send(correlation_data[oid], dest=0, tag=oid)
        
        if verbose:
            print('---> Distributing done </barrier>')

    matches_cdat = []

    if MPI.COMM_WORLD.rank == 0 or not MPI_on:
        if verbose:
            print('Finding best matches.')
        
        match_metric = np.empty((len(correlation_data), 2), dtype = np.float64)
        key_list = []
        key_cnt = 0
        for key, cdats in correlation_data.items():
            key_list.append(key)
            tmp_metrics = np.empty((len(cdats),), dtype=np.float64)
            for cind, cdat in enumerate(cdats):
                tmp_metrics[cind] = metric(t, r, v, cdat['r_ref'], cdat['v_ref'])
            tmp_match = np.argmin(tmp_metrics)

            match_metric[key_cnt, 0] = tmp_metrics[tmp_match]
            match_metric[key_cnt, 1] = tmp_match
            if verbose:
                print('{}: {} metric'.format(key, match_metric[key_cnt]))
            key_cnt += 1


        match = np.argmin(match_metric[:,0])
        

        if len(match_metric) > n_closest:
            all_match = np.argpartition(match_metric[:,0], n_closest)
        else:
            all_match = list(range(len(correlation_data)))

        if verbose:
            print('Best match {}:{} at {} match metric'.format(
                key_list[match],
                int(match_metric[match,1]),
                match_metric[match,0],
            ))
        cdat = correlation_data[key_list[match]][int(match_metric[match,1])]

        r_ref = cdat['r_ref']
        v_ref = cdat['v_ref']

        if out_file is not None:

            with h5py.File(out_file, 'w') as h_corr:
                for key, cdat in correlation_data.items():
                    for dat_key, dat in cdat.items():
                        h_corr[key+'/'+dat_key] = np.array(dat)

                h_corr.attrs['sat_match'] = key_list[match]
                h_corr['residuals'] = np.array(cdat['stat'])

        if len(match_metric) > n_closest:
            for may_match in all_match[:n_closest]:
                matches_cdat.append(correlation_data[key_list[may_match]][int(match_metric[may_match,1])])
        else:
            for may_match in all_match:
                matches_cdat.append(correlation_data[key_list[may_match]][int(match_metric[may_match,1])])
    if step > 1:
        matches_cdat = MPI.COMM_WORLD.bcast(matches_cdat, root=0)

    return matches_cdat


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
    ax.set_title('SAT ID {}'.format(cdat['sat_id']))

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
    ax.set_title('Residuals range: mu={:.3f} km, std={:.3f} km'.format(cdat['stat'][0]*1e-3, cdat['stat'][1]*1e-3))

    ax = fig.add_subplot(222)
    ax.hist((v_ref - v)*1e-3)
    ax.set_xlabel('Velocity residuals [km/s]')
    ax.set_title('Residuals range rate: mu={:.3f} km/s, std={:.3f} km/s'.format(cdat['stat'][2]*1e-3, cdat['stat'][3]*1e-3))
    
    ax = fig.add_subplot(223)
    ax.plot(t - t[0], (r_ref - r)*1e-3)
    ax.set_ylabel('Range residuals [km]')
    ax.set_xlabel('Time [s]')

    ax = fig.add_subplot(224)
    ax.plot(t - t[0], (v_ref - v)*1e-3)
    ax.set_ylabel('Velocity residuals [km/s]')
    ax.set_xlabel('Time [s]')


if __name__ == '__main__':
    import population_library
    import propagator_sgp4
    import radar_library as rlib
    import radar_config

    radar = rlib.eiscat_uhf()

    measurement_folder = './data/uhf_test_data/events'
    tle_file = './data/uhf_test_data/tle-201801.txt'
    measurement_file = measurement_folder + '/det-000000.h5'

    with h5py.File(measurement_file,'r') as h_det:
        r = h_det['r'].value*1e3
        t = h_det['t'].value
        v = -h_det['v'].value #i think juha has wrong sign in analysis

        dat = {
            'r': r,
            't': t,
            'v': v,
        }

    print('Loading population.')
    pop = population_library.tle_snapshot(tle_file, sgp4_propagation=True)

    #pop.filter('oid', lambda oid: np.abs(oid - 43075) < 4)
    print('population size: {}'.format(len(pop)))

    cdat = correlate(
        data = dat,
        station = radar._rx[0],
        population = pop,
        metric = residual_distribution_metric,
        n_closest = 2,
        out_file = None,
        verbose = True,
        MPI_on = False,
    )

    if MPI.COMM_WORLD.rank == 0:
        plot_correlation(dat, cdat[0])
        plot_correlation(dat, cdat[1])
        plt.show()
    