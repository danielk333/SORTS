#!/usr/bin/env python

''' Estimating orbit determination errors. '''

import pathlib

import numpy as np
from tqdm import tqdm

from sorts import interpolation
from .radar.measurement_errors.linearized_coded import LinearizedCodedIonospheric
from .radar.measurement_errors.linearized_coded import LinearizedCoded

def orbit_determination_covariance(
        passes, 
        radar_states,
        space_object,
        variables = ['x','y','z','vx','vy','vz','A'],
        deltas = [1e-4]*3 + [1e-6]*3 + [1e-2],
        seed=123,
        cache_folder=None, 
        ray_bending=True,
        prior_cov_inv=None,
        transforms = {
            'A': (lambda A: np.log10(A), lambda Ainv: 10.0**Ainv),
        },
        interpolator=interpolation.Linear, 
        max_dpos=10e3,
        exact=True,
        logger=None,
        profiler=None,
        interrupt=False,
        parallelization=True,
        n_processes=16,
    ):
    ''' Takes a series of passes and calculates a orbit determination covariance of the measurement model is 
    linear and linearized coded errors with optionally ionospheric effects are included.

    Parameters
    ----------
    passes : list of :class:`sorts.Pass<sorts.radar.passes.Pass>`
        List of radar passes over which the orbit determination covariance.
    radar_states : :class:`sorts.radar.radar_controls.RadarControls`
        Radar states during the measurement. Radar states are a :class:`sorts.radar.radar_controls.RadarControls` object
        which has been generated using the :attr:`sorts.radar.system.radar.Radar.control` method of the radar system performing
        the measurements.
    space_object : :class:`sorts.targets.space_object.SpaceObject`
        :class:`sorts.targets.space_object.SpaceObject` instance being observed by the radar system.
    variables : list of str
        list of space object variable names for which we want to compte the Jacobian for. 
    deltas : list of float
        List of step-sizes used to compute the partial derivatives. The number of elements in ``deltas`` must be
        the same as the number of elements in ``variables``.
    cache_folder : str, default=None
        Path to the folder where the computation results.
    ray_bending : str, default=True
        Use e
    prior_cov_inv : str, default=None
        e
    transforms : dict, default={}
        Transformations (i.e. functions) applied to each variable in ``variables``.
    tx_indices : list of int, default=None
        List of transmitting station indices which measurements will be simulated. If None, the measurements will be simulated
        over all stations
    rx_indices : list of int, default=None
        List of receiving station indices which measurements will be simulated. If None, the measurements will be simulated
        over all stations
    epoch : float, default=None
        Reference simulation epch. If ``None``, the simulation epoch will be the same as the space object by default.
    calculate_snr : bool, default=True 
        If ``True``, the algorithm will compute the SNR measurements for each time points given by ``t_dirs``.
    doppler_spread_integrated_snr : bool, default=False
        If ``True``, the algorithm will compute the incoherent SNR measurements for each time points given by ``t_dirs``.
    snr_limit : bool, default=False
        If ``True``, SNR is lower than :attr:`RX.min_SNRdb<sorts.radar.system.RX.min_SNRdb>` will
        be considered to be 0.0 (i.e. no detection).
    exact : bool, default=False
        If True, the states will be propagated at each time point ``t``, if not, they will be propagated to 
        meet the condition set by ``max_dpos``.
    interpolator : :class:`sorts.Interpolator<sorts.common.interpolation.Interpolator>`, default=:class:`sorts.interpolation.Legendre8<sorts.common.interpolation.Legendre8>`
        Interpolator used to interpolate space object states for each measurement point.
    max_dpos : float, default=100e3
        Maximum distance between two consecutive space object states if propagation is needed.

        .. note::
            Lowering this distance will increase the number of states, therefore increasing computation
            time and RAM usage.
        
    save_states : bool, default=False
        If true, the states of the space object will be saved in the measurement data structure.
    profiler : :class:`sorts.Profiler<sorts.common.profiling.Profiler>`, default=None
        Profiler instance used to monitor computation performances.
    logger : :class:`logging.Logger`, default=None
        Logger instance used to log the status of the computations.
    interrupt : bool, default=False
        If ``True``, the measurement simulator will evalate the stop condition (defined within 
        the :attr:`sorts.Measurement.stop_condition<sorts.radar.measurements.measurement.Measurement.stop_condition>`)
        The simulation will stop at the first time step where the condition is satisfied and return the results of the
        previous steps.

        .. note::
            The default implementation of sorts does not provide any implementation for the ``stop_condition`` method.
            Therefore, to use the stop_condition feature, it is necessary to create a new :class:`Measurement` class 
            inherited from the first :class:`Measurement` class and provide a custom implementation satisfying the
            requirements of the project.

    parallelization : bool, default=True
        If ``True``, the computations will be parallelized over multiple processes using the Python
        **multiprocessing** features.
    n_processes : int, default=16
        Number of processes on which the computations will be run.

    Returns 
    -------
    
    '''
    radar = radar_states.radar

    if cache_folder is not None:
        cache_folder = pathlib.Path(cache_folder)

    #observe all the passes, including measurement Jacobian
    datas = []
    J = None
    for ind, ps in enumerate(passes):
        txi, rxi = ps.station_id

        #Now we load the error model
        if ray_bending:
            err = LinearizedCodedIonospheric(radar.tx[txi], cache_folder=cache_folder, seed=seed)
        else:
            err = LinearizedCoded(radar.tx[txi], cache_folder=cache_folder, seed=seed)

        #the Jacobean is stacked as [r_measurements, v_measurements]^T so we stack the measurement covariance equally
        data, J_rx = radar.measurement_class.compute_measurement_jacobian(
            ps, 
            radar_states=radar_states,
            space_object=space_object, 
            variables=variables, 
            deltas=deltas, 
            snr_limit=True,
            transforms = transforms,
            interpolator=interpolator, 
            max_dpos=max_dpos,
            exact=exact,
            logger=logger,
            profiler=profiler,
            interrupt=interrupt,
            parallelization=parallelization,
            n_processes=n_processes,
        )
        if data is None:
            continue

        datas += [data]

        #now we get the expected standard deviations
        if ray_bending:
            r_stds_tx = err.range_std(data['measurements']['range'], data['measurements']['snr'])
        else:
            r_stds_tx = err.range_std(data['measurements']['snr'])
        v_stds_tx = err.range_rate_std(data['measurements']['snr'])

        #Assume uncorrelated errors = diagonal covariance matrix
        Sigma_m_diag_tx = np.r_[r_stds_tx**2, v_stds_tx**2]

        #we simply append the results on top of each other for each station
        if J is not None:
            J = np.append(J, J_rx, axis=0)
            Sigma_m_diag = np.append(Sigma_m_diag, Sigma_m_diag_tx, axis=0)
        else:
            J = J_rx
            Sigma_m_diag = Sigma_m_diag_tx

    #This means that no passes were observable
    if J is None:
        return None, None

    #diagonal matrix inverse is just element wise inverse of the diagonal
    Sigma_m_inv = np.diag(1.0/Sigma_m_diag)
    #For a thorough derivation of this formula:
    #see Fisher Information Matrix of a MLE with Gaussian errors and a Linearized measurement model

    # print("Sigma_m_inv1=np.array(", np.array2string(Sigma_m_inv, separator=","), ")")
    # print("Sigma_m_diag1=np.array(", np.array2string(Sigma_m_diag, separator=","), ")")
    # print("prior_cov_inv1=np.array(", np.array2string(prior_cov_inv, separator=","), ")")
    # print("J1=np.array(", np.array2string(J, separator=","), ")")
    # print("inv1=np.array(", np.array2string(np.transpose(J).dot(Sigma_m_inv).dot(J), separator=","), ")")
    # print("i1=np.array(", np.array2string(np.linalg.inv(np.transpose(J).dot(Sigma_m_inv).dot(J) + prior_cov_inv), separator=","), ")")

    # TODO verify with old version of sorts, all matrices are identical before inversion 
    # but not after ! 
    if prior_cov_inv is not None:
        Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J + prior_cov_inv)
    else:
        Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J)

    # print("Sigma_orb1=np.array(", np.array2string(Sigma_orb, separator=","), ")")
    return Sigma_orb, datas


def covariance_propagation(
        space_object, 
        orbit_cov, 
        t, 
        variables, 
        samples=100, 
        perturbation_cov=None, 
        perturbed_variables=None, 
        transforms = {
            'A': (lambda A: np.log10(A), lambda Ainv: 10.0**Ainv),
        },
    ):
    ''' 
    Propagate error covariance in time. The time vector should start at the moment of the epoch for the covariance matrix.
    Sample mean position and velocity error.
    Optionally add additional errors.
    '''
    ecef0 = space_object.get_state(t)

    r_diff = np.zeros(len(t), dtype=np.float64)
    v_diff = np.zeros(len(t), dtype=np.float64)

    q = tqdm(total=samples)
    for i in range(samples):
        deltas = np.random.multivariate_normal(np.zeros(orbit_cov.shape[0]),orbit_cov)

        if perturbation_cov is not None:
            pert = np.random.multivariate_normal(np.zeros(perturbation_cov.shape[0]),perturbation_cov)

        dso = space_object.copy()
        
        update = {}
        for ind, var in enumerate(variables):
            if var in transforms:
                Tx = transforms[var][0](getattr(dso, var)) + deltas[ind]
                dx = transforms[var][1](Tx)
            else:
                dx = getattr(dso, var) + deltas[ind]

            update[var] = dx

        if perturbation_cov is not None:
            for ind, var in enumerate(perturbed_variables):
                if var in update:
                    base = update[var]
                else:
                    base = getattr(dso, var)

                if var in transforms:
                    Tx = transforms[var][0](base) + pert[ind]
                    dx = transforms[var][1](Tx)
                else:
                    dx = base + pert[ind]
            
            update[var] = dx

        dso.update(**update)

        ecef1 = dso.get_state(t)

        r_diff += np.linalg.norm(ecef1[3:,:]-ecef0[3:,:], axis=0)
        v_diff += np.linalg.norm(ecef1[3:,:]-ecef0[3:,:], axis=0)

        q.update(1)
        q.set_description(f'covariance_propagation [vars={variables}] : MC Sample n={i}')
    q.close()

    r_diff_stdev = r_diff/samples
    v_diff_stdev = v_diff/samples

    return r_diff_stdev, v_diff_stdev

