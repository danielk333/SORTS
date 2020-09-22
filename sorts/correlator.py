#!/usr/bin/env python

'''Correlate measurement time series with a population of objects to find the best match.

Currently only works for Mono-static measurements.

# TODO: Assume a uniform prior distribution over population index, posterior distribution is the probability of what object generated the data. Probability comes from measurement covariance.
'''

import sys
import os
import time
import glob
from tqdm import tqdm

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    comm = None

import numpy as np
import scipy
import h5py
import scipy.constants as constants



def residual_distribution_metric(t, r, v, r_ref, v_ref):
    '''Using the simulated and the measured ranges and rage-rates calculate a de-correlation metric.
    
    :param numpy.ndarray t: Times in seconds corresponding to measurement and object simulated data.
    :param numpy.ndarray r: Measured ranges in meters
    :param numpy.ndarray v: Measured rage-rates in meters per second
    :param numpy.ndarray r_ref: Object simulated ranges in meters
    :param numpy.ndarray v_ref: Object simulated rage-rates in meters per second
    :return: Metric value, smaller values indicate better match.
    :rtype: float
    '''
    residual_r_mu = np.mean(r_ref - r)
    residual_v_mu = np.mean(v_ref - v)
    
    metric = np.abs(residual_r_mu) + np.abs(residual_v_mu)

    return metric

def generate_measurements(state_ecef, rx_ecef, tx_ecef):

    r_tx = tx_ecef[:,None] - state_ecef[:3,:]
    r_rx = rx_ecef[:,None] - state_ecef[:3,:]

    r_tx_n = np.linalg.norm(r_tx, axis=0)
    r_rx_n = np.linalg.norm(r_rx, axis=0)
    
    r_sim = r_tx_n + r_rx_n
    
    v_tx = -np.sum(r_tx*state_ecef[3:,:], axis=0)/r_tx_n
    v_rx = -np.sum(r_rx*state_ecef[3:,:], axis=0)/r_rx_n

    v_sim = v_tx + v_rx

    return r_sim, v_sim


def correlate(
        measurements, 
        population, 
        metric=residual_distribution_metric, 
        metric_reduce=lambda x,y: x+y,
        n_closest = 1, 
        profiler=None, 
        logger=None, 
        MPI=False, 
        propagator_lighttime_correction=None,
    ):
    '''Given a mono-static measurement of ranges and rage-rates, a radar model and a population: correlate measurements with population.

    :param list data: List of dictionaries that contains measurement data. Contents are described below.
    :param station station: Model of receiver station that performed the measurement.
    :param Population population: Population to correlate against.
    :param function metric: Metric used to correlate measurement and simulated object measurement.
    :param Profiler profiler: 
    :param logging.Logger logger: 
    :param bool MPI: If True use internal parallelization with MPI to calculate correlation. Turn to False to externally parallelize with MPI.
    
    **Measurement data:**

      Each entry in the input :code:`measurements` list must be a dictionary that contains the following fields:
        * 't': [numpy.ndarray] Times relative epoch in seconds
        * 'r': [numpy.ndarray] Two-way ranges in meters
        * 'v': [numpy.ndarray] Two-way range-rates in meters per second
        * 'epoch': [astropy.Time] epoch for measurements
        * 'tx': [sorts.TX] Pointer to the TX station
        * 'rx': [sorts.RX] Pointer to the RX station

    Optionally, if `propagator_lighttime_correction` should be applied, one can supply which index in the list is the TX-TX pair,
    assuming that the time `t` given is the transmission time.

    '''
    
    correlation_data = [None]*len(population)
    match_pop = np.empty((len(population),), dtype=np.float64)


    if MPI and comm is not None:
        step = comm.size
        next_check = comm.rank
    else:
        step = 1
        next_check = 0

    pbars = []
    for pbar_id in range(step):
        pbars.append(tqdm(range(len(population)//step), ncols=100))
    pbar = pbars[next_check]

    for ind, obj in enumerate(population):
        if ind != next_check:
            logger.debug('skipping {}/{}'.format(ind+1, len(population)))
            continue

        pbar.set_description('Correlating object {} '.format(ind))
        pbar.update(1)

        if logger is not None:
            logger.debug('correlating {}/{}'.format(ind+1, len(population)))
            logger.debug(obj)

        correlation_data[ind] = []

        for di, data in enumerate(measurements):
            r = data['r']
            t = data['t']
            v = data['v']
            tx = data['tx']
            rx = data['rx']
            epoch = data['epoch']

            t_sort = t.argsort()
            t = t[t_sort]
            r = r[t_sort]
            v = v[t_sort]

            if propagator_lighttime_correction is not None:
                tx_r = measurements[propagator_lighttime_correction]['r']*0.5
                lt_correction = tx_r/constants.c
                t += lt_correction

            tx_ecef = tx.ecef.copy()
            tx_ecef_norm = tx_ecef/np.linalg.norm(tx_ecef)
            rx_ecef = rx.ecef.copy()
            rx_ecef_norm = rx_ecef/np.linalg.norm(rx_ecef)

            t_prop = (epoch - obj.epoch).sec + t

            states = obj.get_state(t_prop)

            r_ref, v_ref = generate_measurements(states, rx_ecef, tx_ecef)

            match = metric(t, r, v, r_ref, v_ref)

            if di == 0:
                match_pop[ind] = match
            else:
                match_pop[ind] = metric_reduce(match_pop[ind], match)

            cdat = {
                'r_ref': r_ref.copy(),
                'v_ref': v_ref.copy(),
                'match': match,
            }

            correlation_data[ind].append(cdat)

        next_check += step


    if MPI and comm is not None and step > 1:

        if comm.rank == 0:
            if logger is not None:
                logger.debug('Receiving all results <barrier>')

            for T in range(1,comm.size):
                for ID in range(T,len(population),comm.size):
                    match_pop[ID] = comm.recv(source=T, tag=ID)
                    if logger is not None:
                        logger.debug('received packet {} from PID{}'.format(ID, T))
        else:
            if logger is not None:
                logger.debug('Distributing all correlation results to process 0 <barrier>')

            for ID in range(comm.rank,len(population),comm.size):
                comm.send(match_pop[ID], dest=0, tag=ID)
        
        if logger is not None:
            logger.debug('---> Distributing done </barrier>')


    if MPI and comm is not None and step > 1:
        if comm.rank == 0:
            best_matches = np.argsort(match_pop)
        else:
            best_matches = None
        best_matches = comm.bcast(best_matches, root=0)
    else:
        best_matches = np.argsort(match_pop)

    if len(best_matches) > n_closest:
        best_matches = best_matches[:n_closest]

    best_cdata = [None]*len(best_matches)

    for cind in range(len(best_cdata)):
        if MPI and comm is not None and step > 1:
            source_rank = best_matches[cind] % comm.size
            best_cdata[cind] = comm.bcast(correlation_data[best_matches[cind]], root=source_rank)
        else:
            best_cdata[cind] = correlation_data[best_matches[cind]]

    return best_matches, match_pop[best_matches], best_cdata


