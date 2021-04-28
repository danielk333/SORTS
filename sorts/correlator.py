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
        forward_model=generate_measurements,
        variables=['r','v'],
        n_closest=1, 
        profiler=None, 
        logger=None, 
        MPI=False, 
    ):
    '''Given a mono-static measurement of ranges and rage-rates, a radar model and a population: correlate measurements with population.

    :param list measurements: List of dictionaries that contains measurement data. Contents are described below.
    :param Population population: Population to correlate against.
    :param function metric: Metric used to correlate measurement and simulated object measurement.
    :param function metric_reduce: Metric used to correlate measurement and simulated object measurement. Can be `None`, in which case each measurement is correlated individually, for this to work the metric also needs to be vectorized.
    :param function forward_model: A pointer to a function that takes in the ecef-state, the rx and tx station ecefs and calculates the observed variables return as a tuple.
    :param list variables: The data variables recorded by the system.
    :param int n_closest: Number of closest matches to save.
    :param Profiler profiler: Profiler instance for checking function performance.
    :param logging.Logger logger: Logger instance for logging the execution of the function.
    :param bool MPI: If True use internal parallelization with MPI to calculate correlation. Turn to False to externally parallelize with MPI.
    
    **Measurement data:**

      Each entry in the input :code:`measurements` list must be a dictionary that contains the following fields:
        * 't': [numpy.ndarray] Times relative epoch in seconds
        * 'epoch': [astropy.Time] epoch for measurements
        * 'tx': [sorts.TX] Pointer to the TX station
        * 'rx': [sorts.RX] Pointer to the RX station

      Then it will contain a entry for each name in the `variables` list. By default this is
        * 'r': [numpy.ndarray] Two-way ranges in meters
        * 'v': [numpy.ndarray] Two-way range-rates in meters per second

    '''
    
    correlation_data = [None]*len(population)
    
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

    match_pop = None

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
        t_sorts = []
        t_selectors = []
        #built unique time vector
        for di, data in enumerate(measurements):
            t = (data['epoch'] - obj.epoch).sec + data['t']
            t_sort = t.argsort()
            t_sorts.append(t_sort)
            if di == 0:
                t_prop0 = t
                t_selectors = np.full(t.shape, di, dtype=np.int)
            else:
                t_prop0 = np.append(t_prop0, t)
                t_selectors = np.append(t_selectors, np.full(t.shape, di, dtype=np.int))

        if ind == 0:
            if metric_reduce is None:
                match_pop = np.empty((len(population),len(t_prop0)), dtype=np.float64)
            else:
                match_pop = np.empty((len(population),), dtype=np.float64)

        t_prop, t_prop_indices = np.unique(t_prop0, return_inverse=True)
        t_prop_args = np.argsort(t_prop)
        
        t_prop_args_i = np.empty_like(t_prop_args)
        t_prop_args_i[t_prop_args] = np.arange(t_prop_args.size)

        #get all states in one go
        states = obj.get_state(t_prop[t_prop_args])

        #correlate with forward model
        for di, data in enumerate(measurements):
            var = tuple(data[x][t_sorts[di]] for x in variables)
            t = (data['epoch'] - obj.epoch).sec + data['t'][t_sorts[di]]
            tx = data['tx']
            rx = data['rx']
            
            tx_ecef = tx.ecef.copy()
            tx_ecef_norm = tx_ecef/np.linalg.norm(tx_ecef)
            rx_ecef = rx.ecef.copy()
            rx_ecef_norm = rx_ecef/np.linalg.norm(rx_ecef)

            states_data = states[:,t_prop_args_i][:,t_prop_indices][:,t_selectors == di]

            ref_var = forward_model(states_data, rx_ecef, tx_ecef)

            match = metric(t, *(var + ref_var))

            if metric_reduce is None:
                match_pop[ind,t_selectors == di] = match
            else:
                if di == 0:
                    match_pop[ind] = match
                else:
                    match_pop[ind] = metric_reduce(match_pop[ind], match)

            cdat = {f'{x}_ref': ref_var[i].copy()  for i,x in enumerate(variables)}
            cdat['match'] = match

            correlation_data[ind].append(cdat)

        next_check += step

    if MPI and comm is not None and step > 1:

        if comm.rank == 0:
            if logger is not None:
                logger.debug('Receiving all results <barrier>')

            for T in range(1,comm.size):
                for ID in range(T,len(population),comm.size):
                    match_pop[ID,...] = comm.recv(source=T, tag=ID)
                    if logger is not None:
                        logger.debug('received packet {} from PID{}'.format(ID, T))
        else:
            if logger is not None:
                logger.debug('Distributing all correlation results to process 0 <barrier>')

            for ID in range(comm.rank,len(population),comm.size):
                comm.send(match_pop[ID,...], dest=0, tag=ID)
        
        if logger is not None:
            logger.debug('---> Distributing done </barrier>')


    if MPI and comm is not None and step > 1:
        if comm.rank == 0:
            best_matches = np.argsort(match_pop, axis=0)
        else:
            best_matches = None
        best_matches = comm.bcast(best_matches, root=0)
    else:
        best_matches = np.argsort(match_pop, axis=0)

    if best_matches.shape[0] > n_closest:
        best_matches = best_matches[:n_closest,...]

    for pbar in pbars:
        pbar.close()

    if metric_reduce is None:
        best_cdata = [None]*best_matches.shape[0]

        for cind in range(best_matches.shape[0]):
            best_cdata[cind] = [None]*best_matches.shape[1]
            for mind in range(best_matches.shape[1]):
                if MPI and comm is not None and step > 1:
                    source_rank = best_matches[cind] % comm.size
                    best_cdata[cind][mind] = comm.bcast(correlation_data[best_matches[cind,mind]], root=source_rank)
                else:
                    best_cdata[cind][mind] = correlation_data[best_matches[cind,mind]]

    else:
        best_cdata = [None]*best_matches.shape[0]

        for cind in range(len(best_cdata)):
            if MPI and comm is not None and step > 1:
                source_rank = best_matches[cind] % comm.size
                best_cdata[cind] = comm.bcast(correlation_data[best_matches[cind]], root=source_rank)
            else:
                best_cdata[cind] = correlation_data[best_matches[cind]]

    if metric_reduce is None:
        best_metrics = np.empty_like(best_matches)
        for mind in range(best_matches.shape[1]):
            best_metrics[:,mind] = match_pop[best_matches[:,mind],mind]
    else:
        best_metrics = match_pop[best_matches]

    return best_matches, best_metrics, best_cdata


