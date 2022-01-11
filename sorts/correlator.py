#!/usr/bin/env python

'''Correlate measurement time series with a population of objects to find the best match.

Currently only works for Mono-static measurements.

# TODO: Assume a uniform prior distribution over population index, posterior distribution is the probability 
of what object generated the data. Probability comes from measurement covariance.
'''

from tqdm import tqdm

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
except ImportError:
    class COMM_WORLD:
        rank = 0
        size = 1
    comm = COMM_WORLD()

import numpy as np


def residual_distribution_metric(t, r, v, r_ref, v_ref, **kwargs):
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

    index_tuple = (slice(None), ) + tuple(None for x in range(len(state_ecef.shape) - 1))

    r_tx = tx_ecef[index_tuple] - state_ecef[:3, :, ...]
    r_rx = rx_ecef[index_tuple] - state_ecef[:3, :, ...]

    r_tx_n = np.linalg.norm(r_tx, axis=0)
    r_rx_n = np.linalg.norm(r_rx, axis=0)
    r_sim = r_tx_n + r_rx_n
    
    v_tx = -np.sum(r_tx*state_ecef[3:, :, ...], axis=0)/r_tx_n
    v_rx = -np.sum(r_rx*state_ecef[3:, :, ...], axis=0)/r_rx_n

    v_sim = v_tx + v_rx

    return r_sim, v_sim


def within_fow(t, states, rx, tx):
    if len(states.shape) > 2:
        size = np.prod(states.shape[2:])
        shape = states.shape
        states.shape = states.shape[:2] + (size, )
        ok = [np.logical_and(
            rx.field_of_view(states[:, :, 0]),
            tx.field_of_view(states[:, :, 0]),
        )]
        for j in range(1, size):
            ok += [np.logical_and(
                rx.field_of_view(states[:, :, j]),
                tx.field_of_view(states[:, :, j]),
            )]
        ok = np.stack(ok, axis=1)
        states.shape = shape
        ok.shape = shape[1:]
    else:
        ok = np.logical_and(
            rx.field_of_view(states),
            tx.field_of_view(states),
        )
    return ok


def default_propagation_handling(obj, t, t_measurement_indices, measurements):
    return obj.get_state(t)


def correlate(
            measurements, 
            population, 
            metric=residual_distribution_metric, 
            metric_reduce=lambda x, y: x+y,
            forward_model=generate_measurements,
            sorting_function=lambda metric: np.argsort(metric, axis=0),
            valid_measurement_checker=within_fow,
            metric_dtype=np.float64,
            metric_fail_value=np.nan,
            propagation_handling=default_propagation_handling,
            variables=['r', 'v'],
            meta_variables=[],
            n_closest=1, 
            scalar_metric=True,
            profiler=None, 
            logger=None, 
            MPI=False, 
        ):
    '''Given a mono-static measurement of ranges and rage-rates, a radar model and a population: correlate measurements with population.

    # TODO: Update docstring
    # TODO: Add FOV check option

    :param list measurements: List of dictionaries that contains measurement data. Contents are described below.
    :param Population population: Population to correlate against.
    :param function metric: Metric used to correlate measurement and simulated object measurement.
    :param function metric_reduce: Metric used to correlate measurement and simulated object measurement. Can be `None`, in which case each measurement is correlated individually, for this to work the metric also needs to be vectorized.
    :param function forward_model: A pointer to a function that takes in the ecef-state, the rx and tx station ecefs and calculates the observed variables return as a tuple.
    :param function sorting_function: A pointer to a sorting function that takes in the metric result array and returns a list of indices indicating the sorting order.
    :param function valid_measurement_checker: A pointer to a function that checks if the measurnment to calculate a metric is valid or not. By default checks if the object is within both stations FOV. The function expects to take (time, states, rx station, tx station).
    :param function propagation_handling: A pointer to a function that handles the propagation of the object to the measurnment point. By defaults simple uses the space object `get_state` method. TODO: add doc reference here to the default function to use as template for modifications.
    :param numpy.dtype metric_dtype: A valid numpy dtype declaration for the metric output array. This allows complex structured results to be processed.
    :param list variables: The data variables recorded by the system. Theses should be in `measurements` and returned by the `forward_model`.
    :param list meta_variables: The data meta variables recorded by the system. These are input as keyword arguments to the metric and does not need to be produced by any model.
    :param int n_closest: Number of closest matches to save.
    :param bool scalar_metric: indicats if the metric returns a scalar or a vector. If `False` the `metric_reduce` is expected to only take one argument to reduce the vectorized results.
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
    
    correlation_data = {}

    if n_closest > len(population):
        raise ValueError(f'Cannot generate n_closest={n_closest} greater than population size {len(population)}')

    iters = []
    for rank in range(comm.size):
        iters.append(list(range(rank, len(population), comm.size)))
    pbar = tqdm(total=len(iters[comm.rank]), position=comm.rank)

    meas_num = 0
    for data in measurements:
        meas_num += len(data['t'])

    if scalar_metric:
        if metric_reduce is None:
            match_pop = np.empty((n_closest + 1, len(measurements)), dtype=metric_dtype)
            index_pop = np.empty((n_closest + 1, len(measurements)), dtype=np.int64)
        else:
            match_pop = np.empty((n_closest + 1,), dtype=metric_dtype)
            index_pop = np.empty((n_closest + 1,), dtype=np.int64)
    else:
        if metric_reduce is None:
            match_pop = np.empty((n_closest + 1, meas_num), dtype=metric_dtype)
            index_pop = np.empty((n_closest + 1, meas_num), dtype=np.int64)
        else:
            match_pop = np.empty((n_closest + 1,), dtype=metric_dtype)
            index_pop = np.empty((n_closest + 1,), dtype=np.int64)
            tmp_match_cache = np.empty((meas_num, ), dtype=metric_dtype)

    # Just pick some reference epoch so we can shift time vectors
    # Then we shift relative the object later when we propagate
    ref_epoch = measurements[0]['epoch']

    # Build unique time vector from all input measurements
    t_sorts = []
    t_selectors = []
    for di, data in enumerate(measurements):
        t = (data['epoch'] - ref_epoch).sec + data['t']
        t_sort = t.argsort()
        t_sorts.append(t_sort)
        if di == 0:
            t_prop0 = t
            t_selectors = np.full(t.shape, di, dtype=np.int)
            t_m_index = np.arange(len(t))
        else:
            t_prop0 = np.append(t_prop0, t)
            t_selectors = np.append(t_selectors, np.full(t.shape, di, dtype=np.int))
            t_m_index = np.append(t_m_index, np.arange(len(t)))

    t_prop_base, t_prop_indices = np.unique(t_prop0, return_inverse=True)
    t_prop_args = np.argsort(t_prop_base)
    
    t_prop_args_i = np.empty_like(t_prop_args)
    t_prop_args_i[t_prop_args] = np.arange(t_prop_args.size)

    t_indices = []
    t_reverse_inds = np.arange(t_prop_args.size)
    t_reverse_inds = t_reverse_inds[t_prop_args_i][t_prop_indices]
    for tii in range(len(t_prop_base)):
        tii_matches = np.argwhere(tii == t_reverse_inds).flatten()
        t_indices.append([(t_selectors[x], t_m_index[x]) for x in tii_matches])

    iteration_counter = 0

    # Iterate trough population
    for ind in iters[comm.rank]:
        obj = population.get_object(ind)

        t_shift = (ref_epoch - obj.epoch).sec
        t_prop = t_prop_base + t_shift

        pbar.set_description('Correlating object {} '.format(ind))
        pbar.update(1)

        if logger is not None:
            logger.debug('correlating {}/{}'.format(ind+1, len(population)))
            logger.debug(obj)

        object_correlation_data = []

        # get all states in one go
        # and mark which time corresponds to which measurnment set and which measurnment in that set
        states = propagation_handling(obj, t_prop[t_prop_args], t_indices, measurements)

        # correlate with forward model
        for di, data in enumerate(measurements):
            t = (data['epoch'] - obj.epoch).sec + data['t'][t_sorts[di]]
            tx = data['tx']
            rx = data['rx']
            
            tx_ecef = tx.ecef.copy()
            # tx_ecef_norm = tx_ecef/np.linalg.norm(tx_ecef)
            rx_ecef = rx.ecef.copy()
            # rx_ecef_norm = rx_ecef/np.linalg.norm(rx_ecef)

            states_data = states[:, t_prop_args_i, ...][:, t_prop_indices, ...][:, t_selectors == di, ...]

            base_valid = valid_measurement_checker(t, states_data, rx, tx)
            if len(base_valid.shape) > 1:
                valid = np.sum(base_valid, axis=tuple(range(1, len(base_valid.shape)))) > 0
            else:
                valid = base_valid

            var = tuple(data[x][t_sorts[di]][valid] for x in variables)
            ref_var = forward_model(states_data[:, valid, ...], rx_ecef, tx_ecef)

            kwvar = {}
            for x in meta_variables:
                try:
                    kwvar[x] = data[x][t_sorts[di]][valid]
                except TypeError:
                    kwvar[x] = data[x]

            match = metric(t[valid], *(var + ref_var), **kwvar)

            if scalar_metric:
                if match.size == 0:
                    match = metric_fail_value
            else:
                _match = np.empty(t.shape, dtype=metric_dtype)
                _match[valid] = match
                _match[np.logical_not(valid)] = metric_fail_value
                match = _match

            # Calculate the metric, scalar or vectorized and using reduction if defined
            if scalar_metric:
                if metric_reduce is None:
                    match_pop[n_closest, di] = match
                else:
                    if di == 0:
                        match_pop[n_closest] = match
                    else:
                        match_pop[n_closest] = metric_reduce(
                            match_pop[n_closest], 
                            match,
                        )
            else:
                if metric_reduce is None:
                    match_pop[n_closest, t_selectors == di] = match
                else:
                    tmp_match_cache[t_selectors == di] = match
                    if di == len(measurements) - 1:
                        match_pop[n_closest] = metric_reduce(tmp_match_cache)
            
            cdat = {}
            for i, x in enumerate(variables):
                _xp = np.full(base_valid.shape, np.nan, dtype=ref_var[i].dtype)
                _xp[valid, ...] = ref_var[i]
                cdat[f'{x}_ref'] = _xp

            cdat['match'] = match
            cdat['valid'] = valid
            cdat['states'] = states_data
            cdat['oid'] = ind

            object_correlation_data.append(cdat)

        # Set population index for current metric values
        index_pop[n_closest, ...] = ind

        # Save current set of correlation data
        correlation_data[ind] = object_correlation_data

        # Get the current best matches for this rank
        if iteration_counter < n_closest:
            match_pop[iteration_counter, ...] = match_pop[n_closest, ...]
            index_pop[iteration_counter, ...] = index_pop[n_closest, ...]
        else:
            sorting_results = sorting_function(match_pop)
            match_pop = np.take_along_axis(match_pop, sorting_results, axis=0)
            index_pop = np.take_along_axis(index_pop, sorting_results, axis=0)

            # Walk trough saved correlation data and free unused
            kept_index_pop = index_pop[:n_closest, ...]
            for oid in list(correlation_data.keys()):
                if not np.any(oid == kept_index_pop):
                    del correlation_data[oid]

        iteration_counter += 1

    pbar.close()

    # Remove the last row used as cache
    match_pop = match_pop[:n_closest, ...]
    index_pop = index_pop[:n_closest, ...]

    # Communicate the best matches with root thread
    #  to gather results and select the best ones
    if MPI and comm.size > 1:

        matches_pop = [None]*comm.size
        indecies_pop = [None]*comm.size

        matches_pop[comm.rank] = match_pop
        indecies_pop[comm.rank] = index_pop

        if comm.rank == 0:
            if logger is not None:
                logger.debug('Receiving all results <barrier>')

            for T in range(1, comm.size):
                matches_pop[T] = comm.recv(source=T, tag=T*10 + 1)
                indecies_pop[T] = comm.recv(source=T, tag=T*10 + 2)
                correlation_data.update(comm.recv(source=T, tag=T*10 + 3))
                if logger is not None:
                    logger.debug(f'Received data from PID{T}')
        else:
            if logger is not None:
                logger.debug('Distributing all results to process 0 <barrier>')

            comm.send(matches_pop[comm.rank], dest=0, tag=comm.rank*10 + 1)
            comm.send(indecies_pop[comm.rank], dest=0, tag=comm.rank*10 + 2)
            comm.send(correlation_data, dest=0, tag=comm.rank*10 + 3)
        
        if logger is not None:
            logger.debug('---> Distributing done </barrier>')

        if comm.rank == 0:
            match_pop = np.concatenate(matches_pop)
            index_pop = np.concatenate(indecies_pop)

            # Resort and filter after reciving the other MPI data
            sorting_results = sorting_function(match_pop)
            match_pop = np.take_along_axis(match_pop, sorting_results[:n_closest, ...], axis=0)
            index_pop = np.take_along_axis(index_pop, sorting_results[:n_closest, ...], axis=0)

            for oid in list(correlation_data.keys()):
                if not np.any(oid == index_pop):
                    del correlation_data[oid]
        else:
            correlation_data = None

        match_pop = comm.bcast(match_pop, root=0)
        index_pop = comm.bcast(index_pop, root=0)

    return index_pop, match_pop, correlation_data
