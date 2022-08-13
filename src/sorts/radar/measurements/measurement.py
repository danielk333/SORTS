import numpy as np
import ctypes
import multiprocessing

import scipy.constants

from .. import signals
from ..passes import Pass, find_simultaneous_passes, equidistant_sampling
from ...common import interpolation 
from sorts import clibsorts

class Measurement(object):
    ''' Simulates radar measurements.

    Each radar system must be associated with a measurement unit (encapsulated by the :class:`Measurement`) to be able to 
    simulate the observations resulting from given control sequence.
    '''
    def __init__(self, logger=None, profiler=None):
        ''' Default class constructor. '''
        pass

    def compute_measurement_jacobian(
        self, 
        txrx_pass,
        radar_states, 
        space_object, 
        variables, 
        deltas, 
        transforms={},
        interpolator=interpolation.Legendre8, 
        max_dpos=100e3,
        exact=True,
        epoch=None,
        calculate_snr=True,
        doppler_spread_integrated_snr=False,
        snr_limit=True, 
        save_states=False,
        logger=None,
        profiler=None,
        interrupt=False,
        parallelization=True,
        n_processes=16,
        **kwargs
        ):
        ''' Computes the measurement Jacobian with respect to a set of variables.
        
        The measurement jacobian is a powerful tool to propagate errors by locally linearizing the measurement function. 
        This linearization is done according to a set of parameters (or ``variables``).

        Parameters
        ----------
        txrx_pass : list of :class:`sorts.Pass<sorts.radar.passes.Pass>`
            Space object :class:`sorts.Pass<sorts.radar.passes.Pass>` over which the computations will be run.
        radar_states : :class:`sorts.RadarControls<sorts.radar.radar_controls.RadarControls>`
            Radar states during the measurement. Radar states are a :class:`sorts.RadarControls<sorts.radar.radar_controls
            .RadarControls>` object which has been generated using the :attr:`sorts.Radar.control<sorts.radar.system.radar.
            Radar.control>` method of the radar system performing the measurements.
        space_object : :class:`sorts.SpaceObject<sorts.targets.space_object.SpaceObject>`
            Space object instance being observed.
        variables : list of str
            list of space object variable names for which we want to compte the Jacobian for. 
        deltas : list of float
            List of step-sizes used to compute the partial derivatives. The number of elements in ``deltas`` must be
            the same as the number of elements in ``variables``.
        transforms : dict, default={}
            Transformations (i.e. functions) applied to each variable in ``variables``.
        interpolator : :class:`sorts.common.interpolation.Interpolator`, default=interpolation.Legendre8
            Interpolation class used to interpolate between propagated space object states.
        epoch : float, default=None
            Reference simulation epoch (MJD format). If None, the space object epoch will be selected by default.
        calculate_snr : bool, default=True 
            If ``True``, the algorithm will compute the SNR measurements for each time points given by ``t_dirs``.

            .. note::
                Disabling SNR calculations greatly improve performances. 
        
        snr_limit : bool, default=True
            If ``True``, SNR values between :attr:`sorts.radar.system.radar.Radar.min_SNRdb` will be discarded.

        doppler_spread_integrated_snr : bool, default=False
            If ``True``, the algorithm will compute the incoherent SNR measurements for each time points given by ``t_dirs``.
        max_dpos : int, default=100e3
            Maximum distance between two conecutive state sampling points (in meters).
        
        logger : :class:`logging.Logger`, default=None,
            :class:`logging.Logger` instance used to log the status of the computations
        profiler : :class:`sorts.Profiler<sorts.profiling.Profiler>`, default=None,
            Profiler instance used to evaluate the computation performances of the function.
        parallelization : bool, default=True
            If true, the computations will be parallelized using python ``multiprocessing`` capabilities.
            Parallelization is only interesting on multi-core CPUs when the number of states is important
            (usually N>100 states)
        n_processes : int, default=16
            Number of parallel processes running the computations. Only used when ``parallelization`` is ``True``
        kwargs : keyword arguments
            Simulation keyword arguments, if needed.
        '''
        if logger is not None:
            logger.debug(f'Measurement:compute_measurement_jacobian: variables={variables}, deltas={deltas}')

        if profiler is not None:
            profiler.start('Measurement:compute_measurement_jacobian')
            profiler.start('Measurement:compute_measurement_jacobian:reference')

        # simulate observations at the point where the jacobian is evaluated
        data0 = self.compute_pass_measurements(
            txrx_pass,
            radar_states, 
            space_object, 
            epoch=epoch, 
            calculate_snr=calculate_snr, 
            doppler_spread_integrated_snr=doppler_spread_integrated_snr,
            snr_limit=snr_limit, 
            interpolator=interpolator, 
            max_dpos=max_dpos,
            exact=exact,
            use_cached_states=False,
            save_states=True,
            profiler=profiler,
            logger=logger,
            interrupt=interrupt,
            parallelization=parallelization,
            n_processes=n_processes,
            **kwargs)
        
        if data0 is None:
            return None, None
        
        # initialize results and compuation arrays
        t       = data0["measurements"]["t_measurements"]

        # get reference values over each period ID
        J       = np.zeros([len(t)*2, len(variables)], dtype=np.float64)

        dkeep   = np.full(t.shape, False, dtype=np.bool)

        keep    = np.full(t.shape, False, dtype=np.bool)
        r       = np.ndarray((len(t), 2), dtype=np.float64)
        r_dot   = np.ndarray((len(t), 2), dtype=np.float64)

        # get results : range/range-rate
        keep[data0["measurements"]['detection']] = True

        r[keep, 0]      = data0["measurements"]['range'][keep]
        r_dot[keep, 0]  = data0["measurements"]['range_rate'][keep]

        if profiler is not None:
            profiler.stop('Measurement:compute_measurement_jacobian:reference')
            
        # estimate partial derivatives for each variable in ``variables``
        for ind, var in enumerate(variables):
            if profiler is not None:
                profiler.start(f'Measurement:compute_measurement_jacobian:d_{var}')
            if logger is not None:
                logger.info(f'Measurement:compute_measurement_jacobian: computing partial derivatives : d_{var}')

            # apply infinitesimal modification of variable value x + dx 
            dso = space_object.copy()
            if var in transforms:
                Tx = transforms[var][0](getattr(dso, var)) + deltas[ind]
                dx = transforms[var][1](Tx)
            else:
                dx = getattr(dso, var) + deltas[ind]
            
            # update orbit
            dso.update(**{var: dx})

            # compute range and range rates
            ddata = self.compute_pass_measurements(
                txrx_pass,
                radar_states, 
                dso, 
                epoch=epoch, 
                calculate_snr=False, 
                doppler_spread_integrated_snr=False,
                snr_limit=False, 
                interpolator=interpolator, 
                max_dpos=max_dpos,
                exact=exact,
                use_cached_states=False, # disable use of cached propagation results
                save_states=True,
                profiler=profiler,
                logger=logger,
                interrupt=interrupt,
                parallelization=parallelization,
                n_processes=n_processes,
                )

            # update values and compute jacobian
            dkeep[ddata["measurements"]['detection']] = True
            keep = np.logical_and(keep, dkeep)

            r[dkeep, 1] = ddata["measurements"]['range']
            r_dot[dkeep, 1] = ddata["measurements"]['range_rate']

            dr = (r[:, 1] - r[:, 0])/deltas[ind]
            dv = (r_dot[:, 1] - r_dot[:, 0])/deltas[ind]

            J[:len(t), ind] = dr
            J[len(t):, ind] = dv

            if profiler is not None:
                profiler.stop(f'Measurement:compute_measurement_jacobian:d_{var}')

        if len(np.where(keep)[0]) == 0:
            J = None
            data0 = None
        else:
            for key in data0["measurements"]:
                if key in ['detection']:
                    continue
                elif isinstance(data0["measurements"][key], np.ndarray):
                    data0["measurements"][key] = data0["measurements"][key][..., keep]
                else:
                    data0["measurements"][key] = [x for ind_, x in enumerate(data0["measurements"][key]) if keep[data0["measurements"]['detection']][ind_]]
            
            data0["measurements"]['detection'] = np.argwhere(keep).flatten()

            Jkeep   = np.full((len(t)*2,), False, dtype=np.bool)
            Jkeep[:len(t)] = keep
            Jkeep[len(t):] = keep
            J = J[Jkeep, :]

        if profiler is not None:
            profiler.stop('Measurement:compute_measurement_jacobian')

        return data0, J


    def compute_space_object_measurements(
        self,
        radar_states, 
        space_object, 
        tx_indices=None,
        rx_indices=None,
        epoch=None, 
        calculate_snr=True, 
        doppler_spread_integrated_snr=False,
        interpolator=interpolation.Legendre8, 
        max_dpos=100e3,
        exact=False,
        snr_limit=True, 
        save_states=False, 
        logger=None,
        profiler=None,
        interrupt=False,
        parallelization=True,
        n_processes=16,
        ):
        ''' Simulates all the measurements of a space object passing over a radar system given 
        a set of radar states during the measurement interval.

        This function is a convenience function which can be used to simulate observations of 
        a space object for a given set of radar states. The observation simulation will be run over the each 
        pass within the ``radar_states`` time array. If ``exact`` is ``False``, the states will 
        be propagated at low time-resolution and then the ``interpolator`` will be used to interpolate 
        between states for each measurement point.

        .. note::
            It is important to note that before simulating the measurements, it is necessary to generate the
            radar states corresponding to the measurement time interval. To generate the states, it is recommended
            to use a controller, and then use the function :attr:`Radar.control<sorts.radar.system.radar.Radar.control>`
            to generate the states corresponding to the controls generated by the controller. 

        The algorithm will find all space object passes over the radar system and compute the measurements over each 
        pass for each (tx/rx) tuple of stations.
        
        Parameters
        ----------
        radar_states : :class:`sorts.radar.radar_controls.RadarControls`
            Radar states during the measurement. Radar states are a :class:`sorts.radar.radar_controls.RadarControls` object
            which has been generated using the :attr:`sorts.radar.system.radar.Radar.control` method of the radar system performing
            the measurements.
        space_object : :class:`sorts.targets.space_object.SpaceObject`
            :class:`sorts.targets.space_object.SpaceObject` instance being observed by the radar system.
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
        data : dict
            Measurement data structure. The measurement data structure is a python dictionnary containing
            the following keys:

            - **pass_data** : list of measurement data structures simulated over each individual tx-rx pass.

            .. note:: 
                See :attr:`Measurement.compute_pass_measurements` for more information about the data structure.

            - **states** : if ``save_states`` is ``True``, **states** will contain the states of the space object being measured.
        '''
        if radar_states.pdirs is None: # no pointing directions provided, so the time array will be the same as the time slice array
            if logger is not None: 
                logger.debug(f'Measurements:compute_space_object_measurements: No pointing directions in current radar states, ignoring pointing directions...')
            t_dirs = lambda period_id : radar_states.t[period_id]
        else:
            t_dirs = lambda period_id : radar_states.pdirs[period_id]["t"]
        
        if profiler is not None: 
            profiler.start('Measurements:compute_space_object_measurements')

        # Checking input station indices
        tx_indices, rx_indices = self.__get_station_indices(tx_indices, rx_indices, radar_states.radar)
        if logger is not None: 
            logger.debug(f'Measurements:compute_space_object_measurements: stations tx={tx_indices}, rx={rx_indices}')

        if profiler is not None: 
            profiler.start('Measurements:compute_space_object_measurements:Initialization')

        # t is always in scheduler relative time
        # t_samp is in space object relative time if there is a scheduler epoch, otherwise it is assumed that the epoch are the same
        # if there is an interpolator it is assumed that interpolation is done in space object relative time
        if profiler is not None: 
            profiler.start('Measurements:compute_space_object_measurements:Initialization:create_sampling_time_array')

        n_periods = radar_states.n_periods
        if epoch is not None:
            dt_epoch = (epoch - space_object.epoch).sec
        else:
            dt_epoch = 0

        # create state sampling time array
        t_states = equidistant_sampling(orbit=space_object.state, start_t=t_dirs(0)[0], end_t=t_dirs(n_periods-1)[-1], max_dpos=max_dpos) + dt_epoch
        t_states = np.append(t_states, t_states[-1] + (t_states[-1] - t_states[-2])) # add an extra point to the propagation array because last time point is not attained

        if profiler is not None: 
            profiler.stop('Measurements:compute_space_object_measurements:Initialization:create_sampling_time_array')
            profiler.start('Measurements:compute_space_object_measurements:Initialization:get_object_states')

        # propagate space object states
        states = space_object.get_state(t_states)

        data = dict()
        data["pass_data"] = []

        exit_flag = False # flag to exit simulation
        for txi in range(len(tx_indices)):
            for rxi in range(len(rx_indices)):
                # get passes
                if profiler is not None: profiler.start('Measurements:compute_space_object_measurements:Initialization:find_passes')
                passes = find_simultaneous_passes(t_states, states, [radar_states.radar.tx[tx_indices[txi]], radar_states.radar.rx[rx_indices[rxi]]])
                if profiler is not None: profiler.stop('Measurements:compute_space_object_measurements:Initialization:find_passes')

                if logger is not None:
                    logger.info(f"found {len(passes)} passes for stations [tx:{tx_indices[txi]}, rx:{rx_indices[rxi]}]")

                if len(passes) > 0:
                    for pass_id, pass_ in enumerate(passes):
                        if logger is not None:
                            logger.info(f"Computing measurement of pass {pass_id} for stations [tx:{tx_indices[txi]}, rx:{rx_indices[rxi]}]")

                        pass_.station_id = [tx_indices[txi], rx_indices[rxi]]

                        # simulate observations over passes
                        data["pass_data"].append(self.compute_pass_measurements(
                            pass_,
                            radar_states, 
                            space_object, 
                            epoch=epoch, 
                            calculate_snr=calculate_snr, 
                            doppler_spread_integrated_snr=doppler_spread_integrated_snr,
                            interpolator=interpolator, 
                            max_dpos=max_dpos,
                            exact=exact,
                            use_cached_states=False,
                            snr_limit=snr_limit, 
                            save_states=save_states, 
                            logger=logger,
                            profiler=profiler,
                            interrupt=interrupt,
                            parallelization=parallelization,
                            n_processes=n_processes,
                        ))

                        if interrupt is True:
                            if data["pass_data"][-1]["interruption_flag"] is True:
                                exit_flag = True
                                if logger is not None:
                                    logger.info("stopping condition reached, exiting simulation")
                                break

                if exit_flag is True:
                    break
            if exit_flag is True:
                    break

        if profiler is not None: 
            profiler.stop('Measurements:compute_space_object_measurements')

        if save_states is True:
            data["states"] = states

        return data
        

    def compute_pass_measurements(
        self,
        txrx_pass,
        radar_states, 
        space_object, 
        epoch=None, 
        calculate_snr=True, 
        doppler_spread_integrated_snr=False,
        snr_limit=True, 
        interpolator=interpolation.Legendre8, 
        max_dpos=100e3,
        use_cached_states=True,
        exact=False,
        save_states=False, 
        profiler=None,
        logger=None,
        interrupt=False,
        parallelization=True,
        n_processes=16,
        ):
        ''' Simulates radar measurements over a specific pass over a radar system.

        This function is a convenience function which can be used to simulate observations over the
        states of a space object passing over a radar station. If the space object states are not saved 
        within the pass data, then the states will be propagated at low time-resolution and then the
        ``interpolator`` will be used to interpolate between states for each measurement point.

        .. note::
            It is important to note that before simulating the measurements, it is necessary to generate the
            radar states corresponding to the measurement time interval. To generate the states, it is recommended
            to use a controller, and then use the function :attr:`Radar.control<sorts.radar.system.radar.Radar.control>`
            to generate the states corresponding to the controls generated by the controller. 

        Parameters
        ----------
        txrx_pass : :class:`sorts.Pass<sorts.radar.passes.Pass>`
            Space object pass over the radar system which measurements will be simulated.
        radar_states : :class:`sorts.radar.radar_controls.RadarControls`
            Radar states during the measurement. Radar states are a :class:`sorts.radar.radar_controls.RadarControls` object
            which has been generated using the :attr:`sorts.radar.system.radar.Radar.control` method of the radar system performing
            the measurements.
        space_object : :class:`sorts.SpaceObject<sorts.targets.space_object.SpaceObject>`
            :class:`sorts.targets.space_object.SpaceObject` instance being observed by the radar system.
        calculate_snr : bool, default=True 
            If ``True``, the algorithm will compute the SNR measurements for each time points given by ``t_dirs``.
        doppler_spread_integrated_snr : bool, default=False
            If ``True``, the algorithm will compute the incoherent SNR measurements for each time points given by ``t_dirs``.
        snr_limit : bool, default=False
            If ``True``, SNR is lower than :attr:`RX.min_SNRdb<sorts.radar.system.RX.min_SNRdb>` will
            be considered to be 0.0 (i.e. no detection).
        interpolator : :class:`sorts.Interpolator<sorts.common.interpolation.Interpolator>`, default=:class:`sorts.interpolation.
            Legendre8<sorts.common.interpolation.Legendre8>` Interpolator used to interpolate space object states for each measurement 
            point.
        max_dpos : float, default=100e3
            Maximum distance between two consecutive space object states if propagation is needed.

            .. note::
                Lowering this distance will increase the number of states, therefore increasing computation
                time and RAM usage.
        exact : bool, default=False
            If True, the states will be propagated at each time point ``t``, if not, they will be propagated to 
            meet the condition set by ``max_dpos``.
        use_cached_states : bool, default=True
            If True, the function will use the space object states cached within the radar pass if available.
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
        data : dict
            Measurement data structure. The measurement data structure is a python dictionnary containing
            the following keys:

            - **measurements** : contains all the default measurement data:
                -   **t** : Time slice array (start time of each time slice) (R,).
                -   **t_measurements** : Measurement time array, contains all the time points where a measurement
                    has been performed (R,).
                -   **snr** : SNR measurement values (3, R).
                -   **range** : Two-way range values (range from Tx->Target->Rx) (3, R).
                -   **range_rate** : Two-way radial velocity values (Range rate from Tx->Target + Target->Rx).
                -   **pointing_direction** : dictionnary containing the pointing direction of the station 
                    during the measurement: \n\n\n
                    ``pointing_direction[station_type]``\n\n\n
                    Where the station type is either 'rx' or 'tx'. Each dictionnary entry contains an 
                    array (3, R) containing the pointing directions of the Tx/Rx station at the measurement time.
                -   **rcs** : Radar-cross-section of the target (R,).
                -   **detection** : at each time point, ``detection`` will be 1 if the SNR is greater than :attr:`RX.
                    min_SNRdb<sorts.radar.system.station.RX.min_SNRdb>` (R,). 

            - **txi** : index of the transmitting station performing the measurements.

            - **rxi** : index of the receiving station performing the measurements.

            - **snr_inch** : if ``doppler_spread_integrated_snr`` is true, **snr_inch** will contain an array of incoherently integrated SNR measurements. The length of the measurement .

            - **states** : if ``save_states`` is ``True``, **states** will contain the states of the space object being measured.
        '''
        # get states from the pass or propagate (enu)
        txi, rxi = txrx_pass.station_id

        # get pdirs and tdirs arrays
        if radar_states.pdirs is None:
            if logger is not None: 
                logger.debug(f'Measurements:compute_pass_measurements: No pointing directions in current radar states, ignoring pointing directions...')
            
            # no pointing directions provided, so the time array will be the same as the time slice array
            t_dirs = lambda period_id : radar_states.t[period_id]
            pdirs = lambda period_id : None
        else:
            t_dirs = lambda period_id : radar_states.pdirs[period_id]["t"]
            pdirs = lambda period_id : radar_states.pdirs[period_id]

        n_periods = len(radar_states.t)

        # get states interpolator
        if not isinstance(interpolator, interpolation.Interpolator):
            if not issubclass(interpolator, interpolation.Interpolator):
                raise TypeError(f"interpolator must be an instance or subclass of {interpolation.Interpolator}.")

            # get enu space object states
            if txrx_pass.enu is not None and use_cached_states is True:
                tx_enu = txrx_pass.enu[0] # tx_enus/rx_enus
                states = radar_states.radar.tx[txi].to_ecef(tx_enu)
                t_states = txrx_pass.t
            else: 
                # propagate states
                if exact is False: # only consider bounds
                    t_states_prop = np.array([t_dirs(0)[0], t_dirs(n_periods-1)[-1]])
                else: # copy full time array for exact propagation
                    t_states_prop = np.ndarray(0, dtype=float)
                    for period_id in range(n_periods):
                        t_states_prop = np.append(t_states_prop, t_dirs(period_id))
                t_states, states = self.get_states(
                    t_states_prop,  
                    space_object, 
                    epoch=epoch, 
                    max_dpos=max_dpos,
                    logger=logger,
                    profiler=profiler,
                    exact=exact,
                )
                del t_states_prop

            # initializes the states interpolator
            state_interpolator = interpolator(states, t_states)
        else:
            state_interpolator = interpolator

        t_sampling_states = np.ndarray(0, dtype=float)
        space_object_states = np.ndarray((6, 0), dtype=float)
        data = []

        # compute measurements for each period id
        # interruption flag is used to stop simulations if the stop_condition is met
        interruption_flag = False
        for period_id in range(n_periods):
            t_tmp = t_dirs(period_id)
            if t_tmp[-1] > txrx_pass.start():
                # interpolate states over subperiod
                tmp_states = state_interpolator.get_state(t_tmp)

                b_sup   = min(txrx_pass.end(), t_tmp[-1])
                b_inf   = max(txrx_pass.start(), t_tmp[0])
                bounds  = np.array([b_inf, b_sup])

                # generate radar states only for the current period id
                radar_properties = dict()
                for station_type in ("tx", "rx"):
                    radar_properties[station_type] = radar_states.property_controls[period_id][station_type]

                if pdirs(period_id) is None:
                    pdirs_tx = None
                    pdirs_rx = None
                else:
                    pdirs_tx = pdirs(period_id)["tx"][txi, 0]
                    pdirs_rx = pdirs(period_id)["rx"][rxi, txi]

                # compute measurements
                data.append(self.parallel_states_measurements(
                    radar_states.t[period_id],
                    radar_states.t_slice[period_id],
                    t_tmp, 
                    radar_states.radar,
                    pdirs_tx, 
                    pdirs_rx, 
                    space_object,
                    tmp_states, 
                    radar_properties, 
                    radar_states.controlled_properties, 
                    txi, 
                    rxi,
                    bounds=bounds,
                    calculate_snr=calculate_snr, 
                    doppler_spread_integrated_snr=doppler_spread_integrated_snr,
                    snr_limit=snr_limit,
                    save_states=save_states, 
                    profiler=profiler,
                    logger=logger,
                    parallelization=parallelization,
                    n_processes=n_processes,
                    interrupt=interrupt,
                ))

                # save states
                if save_states is True:
                    space_object_states = np.append(space_object_states, tmp_states[:, np.logical_and(t_tmp>=bounds[0], t_tmp<=bounds[1])], axis=1)

                if t_tmp[-1] > txrx_pass.end():
                    break

                # check if stop condition was met
                if interrupt is True:
                    if data[-1]["interruption_flag"] is True:
                        if logger is not None:
                            logger.info("Measurements:compute_pass_measurements: stop condition triggered, exiting simulation.")
                        interruption_flag = True
                        break

        data_final = recover_data(txi, rxi, data, calculate_snr, doppler_spread_integrated_snr)
        if save_states is True:
            data_final["states"] = space_object_states
        if interrupt is True:
            data_final["interruption_flag"] = interruption_flag

        return data_final


    def parallel_states_measurements(
        self,
        t,
        t_slice,
        t_dirs, 
        radar,
        pdirs_tx, 
        pdirs_rx, 
        space_object,
        space_object_states, 
        radar_properties, 
        controlled_properties, 
        txi, 
        rxi,
        bounds=None,
        calculate_snr=True, 
        doppler_spread_integrated_snr=False,
        snr_limit=False,
        save_states=False, 
        profiler=None,
        logger=None,
        interrupt=False,
        parallelization=True,
        n_processes=16,
        ):
        ''' Simulates an observation over a single control period. 

        This function can be used to parallelize the measurement simulation over a set of 
        space object states (when ``parallelization=True``). The algorithm dispaches the
        computations amongst each available process to reduce the computation time when multiple
        CPUs are available.

        .. note:: 
            This function only shows increase in performances when the number of states is 
            important (N> 50 time points). Simulations with low number of states should be
            run in series. 

        Parameters
        ----------
        t : numpy.ndarray (N,)
            Time slice starting points (in seconds).
            Each time slice corresponds to a time period when the radar is being controlled.
        t_slice : numpy.ndarray (N,)
            Duration of each time slice (in seconds).
        t_dirs : numpy.ndarray (M,)
            Pointing direction time array (in seconds).
            Corresponds to the time points when the radar is set to point in a direction given by
            ``pdirs_tx`` and ``pdirs_rx``. 
            This array must also correspond to the object states time.
        radar : :class:`sorts.Radar<sorts.radar.system.radar.Radar>`
            Radar instance performing the measurement.
        pdirs_tx : numpy.ndarray (M,)
            Transmitting station pointing directions in the ECEF coordinate frame. Each element of
            ``pdirs_tx`` gives the pointing direction of the station at the time given by the corresponding
            element of ``t_dirs``.
        pdirs_rx : numpy.ndarray (M,)
            Receiving station pointing directions in the ECEF coordinate frame. Each element of
            ``pdirs_rx`` gives the pointing direction of the station at the time given by the corresponding
            element of ``t_dirs``.
        space_object : :class:`sorts.SpaceObject<sorts.target.space_object.SpaceObject>`
            Space object being observed by the radar system.
        space_object_states : numpy.ndarray (M,)
            States of the space object is the ECEF reference frame.
        radar_properties : dict
            Structure containing all the radar properties as a function of time. This structure must follow the 
            standard:
            
            radar_properties[*station_type*][*property_name*][*station_id*]

            Where: 
             - *station_type* is either "tx" or "rx" depending on the type of station which property values 
             we want to get.
             - *property_name* is the name of the property. The properties of each station can be known by
             calling 

             >>> station.PROPERTIES
             ["wavelength", "n_ipp", ...]

             .. seealso:: 
                - :class:`sorts.Station<sorts.radar.system.station.Station>`
                - :class:`sorts.RX<sorts.radar.system.station.RX>`
                - :class:`sorts.TX<sorts.radar.system.station.TX>`

             ``radar_properties[station_type][property_name]`` is an array of lists of size the number of stations
             of type ``station_type``. 
            - *station_id* is the index of the station of type *station_type* in the radar system. 
            ``radar_properties[station_type][property_name][station_id]`` is a numpy.ndarray (N,) of 
            floats containing all the values of the properties at time ``t``.
        controlled_properties : dict
            Structure containing the list of properties controlled for each station of the radar.
            The structure must follow the standard:

            controlled_properties[*station_type*][*station_id*]
            
            Where:
            - *station_type* is either "tx" or "rx" depending on the type of station which property list 
             we want to get.
            - *station_id* is the index of the station of type *station_type* in the radar system. 
            ``controlled_properties[station_type][property_name][station_id]`` is a list of 
            names of the properties being controlled for the station *station_id*.
        txi : int
            Index of the transmitting station in the radar system.
        rxi : int
            Index of the receiving station in the radar system.
        bounds : numpy.ndarray (2,), default=None
            If provided, ``bounds=(t_start, t_end)`` limits the computations to the time interval specified
            by ``bounds``.
            This is for example useful to only perform computations on the states within a specific :class:`pass<sorts
            .radar.passes.Pass>`.
        calculate_snr : bool, default=True 
            If ``True``, the algorithm will compute the SNR measurements for each time points given by ``t_dirs``.
        doppler_spread_integrated_snr : bool, default=False
            If ``True``, the algorithm will compute the incoherent SNR measurements for each time points given by ``t_dirs``.
        snr_limit : bool, default=False
            If ``True``, SNR is lower than :attr:`RX.min_SNRdb<sorts.radar.system.RX.min_SNRdb>` will
            be considered to be 0.0 (i.e. no detection).
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
        data : dict
            Measurement data structure. The measurement data structure is a python dictionnary containing
            the following keys:

            - **measurements** : contains all the default measurement data:
                -   **t** : Time slice array (start time of each time slice) (R,).
                -   **t_measurements** : Measurement time array, contains all the time points where a measurement
                    has been performed (R,).
                -   **snr** : SNR measurement values (3, R).
                -   **range** : Two-way range values (range from Tx->Target->Rx) (3, R).
                -   **range_rate** : Two-way radial velocity values (Range rate from Tx->Target + Target->Rx).
                -   **pointing_direction** : dictionnary containing the pointing direction of the station 
                    during the measurement: \n\n\n
                    ``pointing_direction[station_type]``\n\n\n
                    Where the station type is either 'rx' or 'tx'. Each dictionnary entry contains an 
                    array (3, R) containing the pointing directions of the Tx/Rx station at the measurement time.
                -   **rcs** : Radar-cross-section of the target (R,).
                -   **detection** : at each time point, ``detection`` will be 1 if the SNR is greater than :attr:`RX.
                    min_SNRdb<sorts.radar.system.station.RX.min_SNRdb>` (R,). 

            - **txi** : index of the transmitting station performing the measurements.

            - **rxi** : index of the receiving station performing the measurements.

            - **snr_inch** : if ``doppler_spread_integrated_snr`` is true, **snr_inch** will contain an array of incoherently integrated SNR measurements. The length of the measurement .

            - **states** : if ``save_states`` is ``True``, **states** will contain the states of the space object being measured.
        '''

        # get object properties
        diameter                    = space_object.d
        spin_period                 = space_object.parameters.get('spin_period', 0.0)
        radar_albedo                = space_object.parameters.get('radar_albedo', 1.0)

        # bounds is used instead of the mask to synchronize the controls and pdirs time arrays
        if bounds is None:
            start_index     = 0
            end_index       = len(t) - 1
            bounds = [t[0], t[-1]]
        else:
            bounds_id       = self.get_bounds(t, bounds)
            start_index     = bounds_id[0]
            end_index       = bounds_id[1]

        # run processes in parallel if activated
        if parallelization is True:
            def process_function(pid, sub_bounds, return_dict):
                ''' subprocess function, computing measurements over a subspace of states. '''
                return_dict[pid] = self.measure_states(
                    t, 
                    t_slice,
                    t_dirs, 
                    space_object_states, 
                    radar,
                    pdirs_tx, 
                    pdirs_rx, 
                    radar_properties, 
                    controlled_properties, 
                    diameter,
                    spin_period,
                    radar_albedo,
                    txi, 
                    rxi,
                    bounds=sub_bounds,
                    calculate_snr=calculate_snr, 
                    doppler_spread_integrated_snr=doppler_spread_integrated_snr,
                    snr_limit=snr_limit, 
                    profiler=profiler,
                    logger=logger,
                    interrupt=False,
                )

            # check stop condition / interruption
            interruption_flag = False
            if interrupt is True:
                interruption_mask = np.asarray(self.stop_condition(
                radar,
                t_dirs, 
                controlled_properties,
                radar_properties,
                space_object_states,
                txi,
                rxi,
                bounds=bounds))

                # interruption_mask[i] is true if stop condition is met at ti
                interrupt_inds = np.where(interruption_mask)[0]
                if len(interrupt_inds) > 0:
                    interruption_flag = True
                    end_index = interrupt_inds[0]-1 # first index where the interruption is reached = end of simulation


            # compute parallelization sub array properties (for control arrays)
            n_time_points = end_index - start_index + 1
           
            # compute number of states per process
            # TODO : look for algorithm to optimally dispatch computations between processes.
            if n_time_points <= n_processes: # if there are less points than processes, reduce number of processes to match number of points
                n_processes = n_time_points
                n_max_points_per_period = 1
                n_points_last_period = 1
            else:
                n_max_points_per_period = int(np.floor(n_time_points/n_processes))
                n_points_last_period = int(n_time_points - (n_processes-1)*n_max_points_per_period)

            # multiprocessing manager
            manager = multiprocessing.Manager()
            process_subgroup = []
            return_dict = manager.dict()
            
            # set up and run computations
            # periods are sliced into ``n_processes`` arrays which are then parallelized
            for pid in range(n_processes):
                if pid < n_processes-1:
                    n_points = n_max_points_per_period
                else:
                    n_points = n_points_last_period

                if n_points > 0:
                    it_controls_start_index = start_index + n_max_points_per_period*pid
                    it_controls_end_index   = it_controls_start_index + n_points - 1

                    sub_bounds = [t[it_controls_start_index], t[it_controls_end_index]]
                    process = multiprocessing.Process(target=process_function, args=(pid, sub_bounds, return_dict)) # create new process
                    process_subgroup.append(process)
                    process.start()

            data = [None]*n_processes

            # wait for each process to be finished
            for pid, process in enumerate(process_subgroup):
                process.join()
                data[pid] = return_dict[pid]
            
            # retreive computation results  
            data_final = recover_data(txi, rxi, data, calculate_snr, doppler_spread_integrated_snr)

            if interrupt is True:
                data_final['interruption_flag'] = interruption_flag
        else: # compute measurements without paralellization
            data_final = self.measure_states(
                t, 
                t_slice,
                t_dirs, 
                space_object_states, 
                radar,
                pdirs_tx, 
                pdirs_rx, 
                radar_properties, 
                controlled_properties, 
                diameter,
                spin_period,
                radar_albedo,
                txi, 
                rxi,
                bounds=bounds,
                calculate_snr=calculate_snr, 
                doppler_spread_integrated_snr=doppler_spread_integrated_snr,
                snr_limit=snr_limit, 
                profiler=profiler,
                logger=logger,
                interrupt=interrupt,
            )

        # save the states of the space object
        if save_states:
            data_final['states'] = space_object_states            

        return data_final


    def get_states(
        self,
        t,
        space_object, 
        epoch=None, 
        max_dpos=50e3,
        exact=False,
        logger=None,
        profiler=None,
    ):
        ''' Propagats the states of the space object at low time-resolution.

        This function propagates the states of the space object at low time resolution which
        can then be used with an :class:`sorts.Interpolator<sorts.common.interpolation.Interpolator>`.

        Parameters
        ----------
        t : numpy.ndarray of float
            Propagation time (in seconds).
        space_object : :class:`sorts.SpaceObject<sorts.targets.space_object.SpaceObject>`
            Space object which states we want to propagate.
        epoch : float, default=None
            Reference epoch of the simulation. If ``None``, the epoch of the space object will be considered 
            by default.
        max_dpos : float, default=100e3
            Maximum distance between two consecutive space object states.

            .. note::
                Lowering this distance will increase the number of states, therefore increasing computation
                time and RAM usage.
        exact : bool, default=False
            If True, the states will be propagated at each time point ``t``, if not, they will be propagated to 
            meet the condition set by ``max_dpos``.
        logger : :class:`logging.Logger`, default=None
            Logger instance used to log the status of the method.
        profile : :class:`sorts.Profiler<sorts.common.profiling.Profiler>`, default=None
            Profiler instance used to monitor the computation performances of the method.

        Returns
        -------
        t_states : numpy.ndarray (N,)
            Time points at which the states were propagated (in seconds).
        states : numpy.ndarray (6, N)
            Space object states :math:`[x, y, z, v_x, v_y, v_z]^T` in the ECEF frame at each time point ``t_states[i]`` 

        '''
        # epoch correction if a custom epoch is provided
        if epoch is not None:
            dt_epoch = (epoch - space_object.epoch).sec
        else:
            dt_epoch = 0

        if profiler is not None: 
            profiler.start('Measurements:Measure:Initialization:create_sampling_time_array')

        # create state sampling time array
        if exact is False:
            t_states = equidistant_sampling(orbit=space_object.state, start_t=t[0], end_t=t[-1], max_dpos=max_dpos) + dt_epoch
        else:
            t_states = t.copy()

        # generate propagation time array
        t_states = np.append(t_states, t_states[-1] + (t_states[-1] - t_states[-2])) # add an extra point to the propagation array because last time point is not attained

        if profiler is not None: 
            profiler.stop('Measurements:Measure:Initialization:create_sampling_time_array')
            profiler.start('Measurements:Measure:Initialization:get_object_states')

        # propagate space object states
        states = space_object.get_state(t_states)

        if profiler is not None: 
            profiler.stop('Measurements:Measure:Initialization:get_object_states')

        return t_states, states


    def measure_states(
        self,
        t, 
        t_slice,
        t_dirs, 
        object_states, 
        radar, 
        pdirs_tx, 
        pdirs_rx, 
        radar_properties, 
        controlled_properties,
        diameter,
        spin_period,
        radar_albedo,
        txi, 
        rxi,
        bounds=None,
        calculate_snr=True, 
        doppler_spread_integrated_snr=False,
        snr_limit=False,
        interrupt=False,
        profiler=None,
        logger=None,
    ):
        ''' Computes the radar measurements given a set of space object states.
        
        This low-level function allows for the computation of radar measurements of 
        a space object given its states over the time period of interrest.

        Parameters
        ----------
        t : numpy.ndarray (N,)
            Time slice starting points (in seconds).
            Each time slice corresponds to a time period when the radar is being controlled.
        t_slice : numpy.ndarray (N,)
            Duration of each time slice (in seconds).
        t_dirs : numpy.ndarray (M,)
            Pointing direction time array (in seconds).
            Corresponds to the time points when the radar is set to point in a direction given by
            ``pdirs_tx`` and ``pdirs_rx``. 
            This array must also correspond to the object states time.
        object_states : numpy.ndarray (M,)
            States of the space object is the ECEF reference frame.
        radar : :class:`sorts.Radar<sorts.radar.system.Radar>`
            Radar instance performing the measurements.
        pdirs_tx : numpy.ndarray (M,)
            Transmitting station pointing directions in the ECEF coordinate frame. Each element of
            ``pdirs_tx`` gives the pointing direction of the station at the time given by the corresponding
            element of ``t_dirs``.
        pdirs_rx : numpy.ndarray (M,)
            Receiving station pointing directions in the ECEF coordinate frame. Each element of
            ``pdirs_rx`` gives the pointing direction of the station at the time given by the corresponding
            element of ``t_dirs``.
        radar_properties : dict
            Structure containing all the radar properties as a function of time. This structure must follow the 
            standard:
            
            radar_properties[*station_type*][*property_name*][*station_id*]

            Where: 
             - *station_type* is either "tx" or "rx" depending on the type of station which property values 
             we want to get.
             - *property_name* is the name of the property. The properties of each station can be known by
             calling 

             >>> station.PROPERTIES
             ["wavelength", "n_ipp", ...]

             .. seealso:: 
                - :class:`sorts.Station<sorts.radar.system.station.Station>`
                - :class:`sorts.RX<sorts.radar.system.station.RX>`
                - :class:`sorts.TX<sorts.radar.system.station.TX>`

             ``radar_properties[station_type][property_name]`` is an array of lists of size the number of stations
             of type ``station_type``. 
            - *station_id* is the index of the station of type *station_type* in the radar system. 
            ``radar_properties[station_type][property_name][station_id]`` is a numpy.ndarray (N,) of 
            floats containing all the values of the properties at time ``t``.
        controlled_properties : dict
            Structure containing the list of properties controlled for each station of the radar.
            The structure must follow the standard:

            controlled_properties[*station_type*][*station_id*]
            
            Where:
            - *station_type* is either "tx" or "rx" depending on the type of station which property list 
             we want to get.
            - *station_id* is the index of the station of type *station_type* in the radar system. 
            ``controlled_properties[station_type][property_name][station_id]`` is a list of 
            names of the properties being controlled for the station *station_id*.
        diameter : float
            Diameter of the :class:`space object<sorts.targets.space_object.SpaceObject>` (in meters).
        spin_period : float
            Spinning period of the :class:`space object<sorts.targets.space_object.SpaceObject>` (in seconds).
            Corresponds to the time needed for the object to complete a full rotation
            around its main axis of rotation.
        radar_albedo : float
            Radar albedo of the :class:`space object<sorts.targets.space_object.SpaceObject>` (-).
            Describes how well the object refects the radar waves in comparison to a perfectly conducting sphere.
        txi : int
            Index of the transmitting station in the radar system.
        rxi : int
            Index of the receiving station in the radar system.
        bounds : numpy.ndarray (2,), default=None
            If provided, ``bounds=(t_start, t_end)`` limits the computations to the time interval specified
            by ``bounds``.
            This is for example useful to only perform computations on the states within a specific :class:`pass<sorts
            .radar.passes.Pass>`.
        calculate_snr : bool, default=True 
            If ``True``, the algorithm will compute the SNR measurements for each time points given by ``t_dirs``.
        doppler_spread_integrated_snr : bool, default=False
            If ``True``, the algorithm will compute the incoherent SNR measurements for each time points given by ``t_dirs``.
        snr_limit : bool, default=False
            If ``True``, SNR is lower than :attr:`RX.min_SNRdb<sorts.radar.system.RX.min_SNRdb>` will
            be considered to be 0.0 (i.e. no detection).
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

        profiler : :class:`sorts.Profiler<sorts.common.profiling.Profiler>`, default=None
            Profiler instance used to monitor computation performances.
        logger : :class:`logging.Logger`, default=None
            Logger instance used to log the status of the computations.
    
        Returns
        -------
        data : dict
            Measurement data structure. The measurement data structure is a python dictionnary containing
            the following keys:

            - **measurements** : contains all the default measurement data:
                -   **t** : Time slice array (start time of each time slice) (R,).
                -   **t_measurements** : Measurement time array, contains all the time points where a measurement
                    has been performed (R,).
                -   **snr** : SNR measurement values (3, R).
                -   **range** : Two-way range values (range from Tx->Target->Rx) (3, R).
                -   **range_rate** : Two-way radial velocity values (Range rate from Tx->Target + Target->Rx).
                -   **pointing_direction** : dictionnary containing the pointing direction of the station 
                    during the measurement: \n\n\n
                    ``pointing_direction[station_type]``\n\n\n
                    Where the station type is either 'rx' or 'tx'. Each dictionnary entry contains an 
                    array (3, R) containing the pointing directions of the Tx/Rx station at the measurement time.
                -   **rcs** : Radar-cross-section of the target (R,).
                -   **detection** : at each time point, ``detection`` will be 1 if the SNR is greater than :attr:`RX.
                    min_SNRdb<sorts.radar.system.station.RX.min_SNRdb>` (R,). 

            - **txi** : index of the transmitting station performing the measurements.

            - **rxi** : index of the receiving station performing the measurements.

            - **snr_inch** : if ``doppler_spread_integrated_snr`` is true, **snr_inch** will contain an array of incoherently integrated SNR measurements. The length of the measurement .

            - **states** : if ``save_states`` is ``True``, **states** will contain the states of the space object being measured.
        '''
        # if the state mask is not provided, define new state mask for measurement computations
        if bounds is None:
            bounds = [t[0], t[-1]]
        
        bound_indices_pdirs = self.get_bounds(t_dirs, bounds)

        # check stop condition / interruption
        interruption_flag = False
        if interrupt is True:
            interruption_mask = np.asarray(self.stop_condition(
                t_dirs, 
                radar,
                controlled_properties,
                radar_properties,
                object_states,
                txi,
                rxi,
                bounds=bounds))

            interrupt_inds = np.where(interruption_mask)[0]
            if len(interrupt_inds) > 0:
                interruption_flag = True

                # update bounds
                bound_indices_pdirs[1] = interrupt_inds[0]-1 # first index where the interruption is reached = end of simulation
                bounds  = [bound_indices_pdirs[0], bound_indices_pdirs[-1]]
        
        # get states which are not observable
        detection_mask = np.full(len(t_dirs), False, bool)
        detection_mask[bound_indices_pdirs[0]:bound_indices_pdirs[-1]+1] = self.observable_filter(
            t_dirs, 
            radar,
            radar_properties,
            controlled_properties,
            object_states,
            txi,
            rxi,
            bounds=bounds,
        )
        n_dirs = len(t_dirs[detection_mask])

        # compute range and range rates
        if profiler is not None: profiler.start('Measurements:Measure:enus,range,range_rate')
        enu_tx_to_so, enu_rx_to_so, ranges_tx, ranges_rx, range_rates_tx, range_rates_rx = self.compute_ranges_and_range_rates(object_states[:, detection_mask], radar, txi, rxi)
        if profiler is not None: profiler.stop('Measurements:Measure:enus,range,range_rate')

        ranges            = ranges_tx + ranges_rx
        range_rates       = range_rates_tx + range_rates_rx

        # compute SNR
        if calculate_snr:
             # compute beam gains
            if profiler is not None: profiler.start('Measurements:Measure:beam_gain')
            gain_tx, gain_rx, wavelength_tx, wavelength_rx = self.compute_gain_and_wavelength(
                t, 
                t_dirs[detection_mask], 
                pdirs_tx[:, detection_mask], 
                pdirs_rx[:, detection_mask], 
                radar, 
                radar_properties, 
                controlled_properties, 
                enu_tx_to_so, 
                enu_rx_to_so, 
                txi, 
                rxi, 
                profiler=profiler,
            )
            if profiler is not None: profiler.stop('Measurements:Measure:beam_gain')

            if profiler is not None: profiler.start('Measurements:Measure:rcs')
            rcs = signals.hard_target_rcs(wavelength_tx, diameter=diameter)       
            if profiler is not None: profiler.stop('Measurements:Measure:rcs')

            if profiler is not None: 
                profiler.start('Measurements:Measure:snr')
            
            # snr data arrays
            snr             = np.ndarray((n_dirs,), dtype=np.float64)
            snr_inch        = np.ndarray((n_dirs,), dtype=np.float64)
            detection       = np.ones((n_dirs,), dtype=np.int32)

            clibsorts.compute_measurement_snr.argtypes = [
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t.ndim, shape=t.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t_dirs[detection_mask].ndim, shape=t_dirs[detection_mask].shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t_slice.ndim, shape=t_slice.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["pulse_length"][txi].ndim, shape=radar_properties["tx"]["pulse_length"][txi].shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["ipp"][txi].ndim, shape=radar_properties["tx"]["ipp"][txi].shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=gain_tx.ndim, shape=gain_tx.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=gain_rx.ndim, shape=gain_rx.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=wavelength_tx.ndim, shape=wavelength_tx.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["power"][txi].ndim, shape=radar_properties["tx"]["power"][txi].shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=ranges_tx.ndim, shape=ranges_tx.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=ranges_rx.ndim, shape=ranges_rx.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["duty_cycle"][txi].ndim, shape=radar_properties["tx"]["duty_cycle"][txi].shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["coh_int_bandwidth"][txi].ndim, shape=radar_properties["tx"]["coh_int_bandwidth"][txi].shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=snr.ndim, shape=snr.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=snr_inch.ndim, shape=snr_inch.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=detection.ndim, shape=detection.shape),
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_double,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_int,
                ]

            clibsorts.compute_measurement_snr(
                t,
                t_dirs[detection_mask],
                t_slice,
                radar_properties["tx"]["pulse_length"][txi],
                radar_properties["tx"]["ipp"][txi],
                gain_tx,
                gain_rx,
                wavelength_tx,
                radar_properties["tx"]["power"][txi],
                ranges_tx,
                ranges_rx,
                radar_properties["tx"]["duty_cycle"][txi],
                radar_properties["tx"]["coh_int_bandwidth"][txi],
                snr,
                snr_inch,
                detection,
                ctypes.c_double(radar.rx[rxi].noise_temperature),
                ctypes.c_double(diameter),
                ctypes.c_double(radar_albedo),
                ctypes.c_double(spin_period),
                ctypes.c_double(radar.min_SNRdb),
                ctypes.c_int(doppler_spread_integrated_snr),
                ctypes.c_int(snr_limit),
                ctypes.c_int(len(t)),
                ctypes.c_int(n_dirs),
            )

            if profiler is not None: 
                profiler.stop('Measurements:Measure:snr')
        else:
            snr             = None
            snr_inch        = None
            rcs             = None
            detection       = None

        # remove pointing directions outside of pass
        pointing_direction = dict()
        if pdirs_tx is not None: 
            pdirs_tx[:, detection_mask]
        pointing_direction["tx"] = pdirs_tx

        if pdirs_rx is not None: 
            pdirs_rx[:, detection_mask]
        pointing_direction["rx"] = pdirs_rx

        # remove control indices outside of the bounds
        bound_indices_ctrl  = self.get_bounds(t, bounds)
        t_tmp = t[bound_indices_ctrl[0]:bound_indices_ctrl[1]+1]

        # change detection array to boolean
        if detection is not None:
            detection = detection.astype(bool)

        # intialize data structure
        data = dict()
        data["measurements"] = dict(
            t=t_tmp,
            t_measurements=t_dirs[detection_mask],
            snr=snr,
            range=ranges,
            range_rx=ranges_rx,
            range_rate=range_rates,
            pointing_direction=pointing_direction,
            rcs=rcs,
            detection=detection)

        data["txi"] = txi
        data["rxi"] = rxi

        if interrupt is True:
            data["interruption_flag"] = interruption_flag
        if doppler_spread_integrated_snr:
            data['snr_inch'] = snr_inch

        return data


    def stop_condition(
        self,
        t, 
        radar,
        radar_properties,
        controlled_properties,
        space_object_states,
        txi,
        rxi,
        bounds=None,
    ):
        ''' Measurement abort/stop condition (i.e. stop time, ...).

        The measurement simulation is stopped when the stop condition is reached. 
        Provide a custom implementation meeting your requirements.

        Parameters
        ----------
        t : numpy.ndarray
            Measurement time (in seconds).
        radar : :class:`sorts.radar.system.radar.Radar`
            :class:`sorts.radar.system.radar.Radar` instance performing the measurements.
        radar_properties : dict
            Structure containing all the radar properties as a function of time. This structure must follow the 
            standard:
            
            radar_properties[*station_type*][*property_name*][*station_id*]

            Where: 
             - *station_type* is either "tx" or "rx" depending on the type of station which property values 
             we want to get.
             - *property_name* is the name of the property. The properties of each station can be known by
             calling 

             >>> station.PROPERTIES
             ["wavelength", "n_ipp", ...]

             .. seealso:: 
                - :class:`sorts.Station<sorts.radar.system.station.Station>`
                - :class:`sorts.RX<sorts.radar.system.station.RX>`
                - :class:`sorts.TX<sorts.radar.system.station.TX>`

             ``radar_properties[station_type][property_name]`` is an array of lists of size the number of stations
             of type ``station_type``. 
            - *station_id* is the index of the station of type *station_type* in the radar system. 
            ``radar_properties[station_type][property_name][station_id]`` is a numpy.ndarray (N,) of 
            floats containing all the values of the properties at time ``t``.
        controlled_properties : dict
            Structure containing the list of properties controlled for each station of the radar.
            The structure must follow the standard:

            controlled_properties[*station_type*][*station_id*]
            
            Where:
            - *station_type* is either "tx" or "rx" depending on the type of station which property list 
             we want to get.
            - *station_id* is the index of the station of type *station_type* in the radar system. 
            ``controlled_properties[station_type][property_name][station_id]`` is a list of 
            names of the properties being controlled for the station *station_id*.
        space_object_states : (6, N)
            States of the space object.
        txi : int
            Index of the transmitting station.
        rxi : int 
            Index of the receiving station
        bounds : list of floats (2,), default=None
            If provided, the stop condition will only be evaluated for time values within ``bounds``.

        Returns
        -------
        bool / numpy.ndarray of bool : 
            ``True`` if the stop condition is reached. If not, ``False``.
        '''
        return False


    def observable_filter(
        self,
        t, 
        radar,
        radar_properties,
        controlled_properties,
        space_object_states,
        txi,
        rxi,
        bounds=None,
    ):
        ''' Determines if the object was observable or not by the radar system.

        This function can be used to define additional observational constraints on the object.
        If those constraints are met, the object states satisfying them will be considered as
        not detected.
        Provide a custom implementation meeting your requirements.

        Parameters
        ----------
        t : numpy.ndarray
            Measurement time (in seconds).
        radar : :class:`sorts.radar.system.radar.Radar`
            :class:`sorts.radar.system.radar.Radar` instance performing the measurements.
        radar_properties : dict
            Structure containing all the radar properties as a function of time. This structure must follow the 
            standard:
            
            radar_properties[*station_type*][*property_name*][*station_id*]

            Where: 
             - *station_type* is either "tx" or "rx" depending on the type of station which property values 
             we want to get.
             - *property_name* is the name of the property. The properties of each station can be known by
             calling 

             >>> station.PROPERTIES
             ["wavelength", "n_ipp", ...]

             .. seealso:: 
                - :class:`sorts.Station<sorts.radar.system.station.Station>`
                - :class:`sorts.RX<sorts.radar.system.station.RX>`
                - :class:`sorts.TX<sorts.radar.system.station.TX>`

             ``radar_properties[station_type][property_name]`` is an array of lists of size the number of stations
             of type ``station_type``. 
            - *station_id* is the index of the station of type *station_type* in the radar system. 
            ``radar_properties[station_type][property_name][station_id]`` is a numpy.ndarray (N,) of 
            floats containing all the values of the properties at time ``t``.
        controlled_properties : dict
            Structure containing the list of properties controlled for each station of the radar.
            The structure must follow the standard:

            controlled_properties[*station_type*][*station_id*]
            
            Where:
            - *station_type* is either "tx" or "rx" depending on the type of station which property list 
             we want to get.
            - *station_id* is the index of the station of type *station_type* in the radar system. 
            ``controlled_properties[station_type][property_name][station_id]`` is a list of 
            names of the properties being controlled for the station *station_id*.
        space_object_states : (6, N)
            States of the space object.
        txi : int
            Index of the transmitting station.
        rxi : int 
            Index of the receiving station
        bounds : list of floats (2,), default=None
            If provided, the stop condition will only be evaluated for time values within ``bounds``.
        Returns
        -------
        bool / numpy.ndarray of bool : 
            ``True`` if the space object is observable. If not, ``False``.
        '''
        return True


    def __get_station_indices(self, tx_indices, rx_indices, radar):
        ''' Gets the station indices from the radar system.

        This function returns the indices of the stations participating in the measurement.
        
        Parameters
        ----------
        tx_indices : list of int
            List of :class:`TX<radar.system.station.TX>` station indices participating in the measurement.
            If ``tx_indices`` is None, then all the :class:`TX<radar.system.station.TX>` stations of the radar system will be 
            used during the measurement.
        rx_indices : list of int
            List of :class:`RX<radar.system.station.RX>` station indices participating in the measurement.
            If  ``rx_indices`` is None, then all the :class:`RX<radar.system.station.RX>` stations of the radar system will be 
            used during the measurement.
        radar : :class:`sorts.radar.system.radar.Radar`
            Radar instance performing the measurement.

        Returns
        -------
        tx_indices : list of int
            Final list of :class:`TX<radar.system.station.TX>` station indices participating in the measurement.
        rx_indices : list of int
            Final list of :class:`RX<radar.system.station.RX>` station indices participating in the measurement.
        '''
        # selecting Rx indices from which measurements are computed
        if rx_indices is None:
            rx_indices = np.arange(0, len(radar.rx))
        else:
            tmp_ids = []
            for id_ in rx_indices:
                if id_ in rx_indices:
                    tmp_ids.append(id_)

            rx_indices = np.asarray(tmp_ids, dtype=int)

        # selecting Tx indices from which measurements are computed
        if tx_indices is None:
            tx_indices = np.arange(0, len(radar.tx))
        else:
            tmp_ids = []
            for id_ in tx_indices:
                if id_ in tx_indices:
                    tmp_ids.append(id_)

            tx_indices = np.asarray(tmp_ids, dtype=int)

        return tx_indices, rx_indices

    def get_beam_gain_and_wavelength(self, beam, enu):
        ''' Computes the gain and wavelength of the radar beam.

        This function computes the ``wavelength`` and the ``gain`` in a the 
        direction of the object in the local station frame of reference (enu)
        given the properties of the station during the measurement.

        To ensure that this function behaves as expected, it is necessary to set
        *all* the properties of the beam **before** trying to compute the gain.

        If ``beam`` has multiple pointing directions, then only the maximum gain 
        will be returned.

        Parameters
        ----------
        beam : :class:`pyant.Beam`
            Gain pattern used to compute the gain and wavelength.
        enu : numpy.ndarray (3,)
            Directions of the target in the station reference frame (ENU coordinates).
            If the beam possesses mutiple pointing directions, then only the maximum 
            gain will be returned.

        Returns
        -------
        gain : float
            Gain (or maximum gain) of the antenna in given direction (linear gain).
        wavelength : float
            Beam wavelength (in meters).
        '''
        if len(beam.pointing.shape) > 1:
            g = np.max([
                beam.gain(enu, ind={'pointing': pi})
                for pi in range(beam.pointing.shape[1])
            ])
        else:
            g = beam.gain(enu)

        wavelength = beam.wavelength
        return g, wavelength


    def compute_gain_and_wavelength(
        self, 
        t,
        t_dirs,
        pdirs_tx, 
        pdirs_rx, 
        radar, 
        property_controls, 
        controlled_properties, 
        tx_enus, 
        rx_enus, 
        txi, 
        rxi, 
        profiler=None
        ):
        ''' Computes the beam **gain** and **wavelength** values for multiple pointing measurement points.

        This function sets the properties of the beam according to the radar station states and then computes
        the gain and wavelength according to the space object position at each measurement time point.  
        
        Parameters
        ----------
        t : numpy.ndarray of float (N,)
            Control/radar states time (in seconds).
        t_dirs : numpy.ndarray of float (M,)
            Measurement time (in seconds).
            The measurement time is the same as the ``pointing_direction`` time and must always have 
            a greater or equal number of elements as `t`.
        pdirs_tx : numpy.ndarray of float (M,)
            :class:`TX<sorts.radar.system.station.TX>` station pointing directions (ECEF coordinate frame).
        pdirs_rx : numpy.ndarray of float
            :class:`RX<sorts.radar.system.station.RX>` station pointing directions (ECEF coordinate frame).
        radar : :class:`sorts.Radar<sorts.radar.system.radar.Radar>`
            :class:`Radar<sorts.radar.system.radar.Radar>` system performing the measurements.
        property_controls : dict
            Property controls of the radar stations. 
            ``property_controls`` must only contain the controls for the radar system performing the measurement **for
            a single control period**.

            .. seealso:: 
                see :attr:`sorts.RadarControls.property_controls<sorts.radar.radar_controls.RadarControls.property_controls>` 
                for more information.

        controlled_properties : dict
            Dictionnary containing the list of all the controlled properties of the radar stations.
            
            ``controlled_properties`` must contain two keys "tx" and "rx". Each dictionnary entry is associated with an array of lists
            of station properties. When there are multiple stations of type "rx", it is possible to access the properties of 
            the ith :class:`RX<sorts.radar.system.station.RX>` by running:

            >>> controlled_properties["rx"][i]
            ["wavelength"]

            .. rubric:: example :

            Consider the EISCAT_3D radar, comprised of 3 :class:`RX<sorts.radar.system.station.RX>` stations and of
            1 :class:`TX<sorts.radar.system.station.TX>` station. By default, ``controlled_properties`` values for 
            the key "rx" have to be:

            >>> controlled_properties["rx"] = [["wavelength"], ["wavelength"], ["wavelength"]]

            The dictionnary entry "rx" contains an array of 3 lists (one per RX station). Each one of those list contains
            the properties of the associated station. For the key "tx", the default values are:

            >>> controlled_properties["tx"] = [["wavelength", "power", "ipp", "n_ipp", "pulse_length", "coh_int_bandwidth", "duty_cycle", "bandwidth"]]

        tx_enus : numpy.ndarray of float (3, M)
            Directions of the target in the transmitting station's frame of reference.
            There must be as many directions as there are measurements.
        rx_enus : numpy.ndarray of float (3, M)
            Directions of the target in the transmitting station's frame of reference.
            There must be as many directions as there are measurements.
        txi : int
            Index of the transmitting station performing the measurement.
        rxi : int
            Index of the transmitting station performing the measurement.
        profiler : :class:`sorts.Profiler<sorts.common.profiling.Profiler>`, default=None
            Profiler instance used to monitor the computation performances. 

        Returns
        -------
        gain_tx : numpy.ndarray of float
            Gain of the transmitting antenna for each measurement point (linear gain).
        gain_rx : numpy.ndarray of float
            Gain of the receiving antenna for each measurement point (linear gain).
        wavelength_tx : numpy.ndarray of float
            Wavelength of the transmitting antenna for each measurement point (in meters).
        wavelength_rx : numpy.ndarray of float
            Wavelength of the receiving antenna for each measurement point (in meters).
        '''
        N = len(t)
        N_dirs = len(t_dirs)
        
        gain_tx = np.zeros((N_dirs,), dtype=np.float64)
        gain_rx = np.zeros((N_dirs,), dtype=np.float64)

        wavelength_tx = np.zeros((N_dirs,), dtype=np.float64)
        wavelength_rx = np.zeros((N_dirs,), dtype=np.float64)
 
        def tx_gain(t_dir, ti):
            ''' callback function to compute the gain of the transmitting station. '''
            nonlocal property_controls, controlled_properties
            station = radar.tx[txi]

            if profiler is not None: 
                profiler.start('Measurements:compute_gain:set_properties')

            # set tx properties
            if pdirs_tx is not None:
                station.point_ecef(pdirs_tx[:, t_dir]) # point the station in the controlled direction
            for property_name in controlled_properties["tx"][txi]: # set each property according to the radar state at ti
                if hasattr(station, property_name):
                    exec("radar.tx[txi]." + property_name + " = property_controls['tx'][property_name][txi][ti]")

            if profiler is not None: 
                profiler.stop('Measurements:compute_gain:set_properties')
                profiler.start('Measurements:compute_gain:gain')

            gain_tx[t_dir], wavelength_tx[t_dir] = self.get_beam_gain_and_wavelength(station.beam, tx_enus[0:3, t_dir])
            
            if profiler is not None: 
                profiler.stop('Measurements:compute_gain:gain')


        def rx_gain(t_dir, ti):
            ''' callback function to compute the gain of the receiving station. '''
            nonlocal property_controls, controlled_properties
            station = radar.rx[rxi]

            if profiler is not None: 
                profiler.start('Measurements:compute_gain:set_properties')

            # set beam properties
            if pdirs_rx is not None:
                station.point_ecef(pdirs_rx[:, t_dir]) # point the station in the controlled direction
            for property_name in controlled_properties["rx"][rxi]: # set each property according to the radar state at ti
                if hasattr(station, property_name):
                    exec("radar.rx[rxi]." + property_name + " = property_controls['rx'][property_name][rxi][ti]")

            if profiler is not None: 
                profiler.stop('Measurements:compute_gain:set_properties')
                profiler.start('Measurements:compute_gain:gain')

            gain_rx[t_dir], wavelength_rx[t_dir] = self.get_beam_gain_and_wavelength(station.beam, rx_enus[0:3, t_dir])

            if profiler is not None: 
                profiler.stop('Measurements:compute_gain:gain')


        # define callback functions 
        COMPUTE_GAIN_TX = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int)
        COMPUTE_GAIN_RX = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int)
        compute_antenna_gain_tx_c = COMPUTE_GAIN_TX(tx_gain)
        compute_antenna_gain_rx_c = COMPUTE_GAIN_RX(rx_gain)

        clibsorts.compute_gain.argtypes = [
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t.ndim, shape=t.shape),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t_dirs.ndim, shape=t_dirs.shape),
            ctypes.c_int,
            ctypes.c_int,
            COMPUTE_GAIN_TX,
            COMPUTE_GAIN_RX,
            ]

        clibsorts.compute_gain(
            t,
            t_dirs,
            ctypes.c_int(N),
            ctypes.c_int(N_dirs),
            compute_antenna_gain_tx_c,
            compute_antenna_gain_rx_c,
            )

        return gain_tx, gain_rx, wavelength_tx, wavelength_rx


    def compute_ranges_and_range_rates(self, states, radar, txi, rxi):
        ''' Computes the range and range rate of the target for each state.

        This function computes the relative range and radial velocity of the target 
        with respect to the radar stations performing the measurement. This 
        computation is performed for each individual target state.

        The function also returns direction of the target with respect to each station
        in its local ENU coordinate frame. 

        .. seealso::
            - :attr:`Pass.calculate_range<sorts.radar.passes.Pass.calculate_range>`
            - :attr:`Pass.calculate_range_rate<sorts.radar.passes.Pass.calculate_range>`

        Parameters
        ----------
            states : numpy.ndarray (6, M)
                States of the space object which properties are measured.
            radar : :class:`sorts.Radar<sorts.radar.system.radar.Radar>`
                Radar system performing the measurements.
            txi : int
                Index of the transmitting station within the ``radar.tx`` list.
            rxi : int
                Index of the receiving station within the ``radar.rx`` list.
    
        Returns
        -------
        enu_tx_to_so : numpy.ndarray  (M,)
            Direction of the target in the local ENU coordinate frame of the transmitting station.
        enu_rx_to_so : numpy.ndarray  (M,)
            Direction of the target in the local ENU coordinate frame of the receiving station.
        ranges_tx : numpy.ndarray  (M,)
            Range of the target with respect to the transmitting station (in meters).
        ranges_rx : numpy.ndarray  (M,)
            Range of the target with respect to the receiving station (in meters).
        range_rates_tx : numpy.ndarray  (M,)
            Radial velocity of the target with respect to the transmitting station (in m/s).
        range_rates_rx : numpy.ndarray  (M,)
            Radial velocity of the target with respect to the receiving station (in m/s).
        '''
        # Transmitters
        enu_tx_to_so     = radar.tx[txi].enu(states)
        ranges_tx        = Pass.calculate_range(enu_tx_to_so)
        range_rates_tx   = Pass.calculate_range_rate(enu_tx_to_so)

        # Receivers
        enu_rx_to_so     = radar.rx[rxi].enu(states)
        ranges_rx        = Pass.calculate_range(enu_rx_to_so)
        range_rates_rx   = Pass.calculate_range_rate(enu_rx_to_so)

        return enu_tx_to_so, enu_rx_to_so, ranges_tx, ranges_rx, range_rates_tx, range_rates_rx


    def get_bounds(self, t, bounds):
        ''' Gets the indices of the bounds of an ordered time array. ''' 
        N = len(t)

        if bounds is None: # if no bounds are provided, return min/max bounds
            return np.array([0, N-1])
        else: # compute bound indices 
            if bounds[0] > bounds[-1]: 
                raise Exception(f"the lower bound must be less than or equal to the upper bound, not {bounds}")

            if bounds[0] <=  t[0]:
                b_inf = 0
            else:
                b_inf = np.argmax(t >= bounds[0])

            if bounds[1] >=  t[-1]:
                b_sup = N-1
            else:
                b_sup = np.argmax(t[b_inf:] > bounds[1]) + b_inf - 1

            return np.array([b_inf, b_sup])



def recover_data(txi, rxi, datas, calculate_snr, doppler_spread_integrated_snr):
    ''' Fuses multiple computation results into a single structure. 

    This function converts an array of measurement data structures of 
    the form ``[data1, data2, data3, ....]`` in a single bigger
    data structure ``data`` containing all the information of 
    the previous data structures.

    .. note::
        The data contained in each sub data structure of ``datas`` will be copied into 
        the new structure following their order of arrival (i.e. ``data1`` will be copied first,
        then ``data2``, and so on).

        Beware that all the measurement data structures in ``datas`` must follow the same
        structure. If ``calculate_snr`` or ``doppler_spread_integrated_snr`` are True, then 
        all the measurements must contain SNR/incoherent SNR measurements.

    Parameters
    ----------
    txi : int
        Index of the transmitting station (within the ``radar.tx`` list) associated with the
        measurement data structures. 
    rxi : int
        Index of the receiving station (within the ``radar.rx`` list) associated with the
        measurement data structures. 
    datas : list
        List of measurement data structures to combine.

        .. seealso::
            See :attr:`Measurement.measure_states` for more information about the output measurement
            data structure.
    calculate_snr : bool
        If True, the data related to SNR measurements will be
        copied in the new structure.  
    doppler_spread_integrated_snr : bool
        If True, the data related to incoherent SNR measurements will be
        copied in the new structure.  

    Returns
    -------
    data : dict
        Measurement data structure containing all the  
    '''

    # initialization of the final measurement data arrays
    t_ctrl              = np.ndarray(0, dtype=float)
    t_measurements      = np.ndarray(0, dtype=float)
    ranges              = np.ndarray(0, dtype=float)
    range_rx            = np.ndarray(0, dtype=float)
    range_rates         = np.ndarray(0, dtype=float)

    pdirs = dict()
    pdirs["tx"]         = np.ndarray((3, 0), dtype=float)
    pdirs["rx"]         = np.ndarray((3, 0), dtype=float)

    if calculate_snr is True:
        snr             = np.ndarray(0, dtype=float)
        rcs             = np.ndarray(0, dtype=float)
        detection       = np.ndarray(0, dtype=bool)

        if doppler_spread_integrated_snr is True:
            snr_inch        = np.ndarray(0, dtype=float)
    else:
        snr             = None
        snr_inch        = None
        rcs             = None
        detection       = None

    # combine all the measurement data from ``datas`` 
    for i, dat in enumerate(datas):
        if dat is not None:
            t_ctrl              = np.append(t_ctrl, dat["measurements"]["t"])
            t_measurements      = np.append(t_measurements, dat["measurements"]["t_measurements"])
            ranges              = np.append(ranges, dat["measurements"]["range"])
            range_rx            = np.append(range_rx, dat["measurements"]["range_rx"])
            range_rates         = np.append(range_rates, dat["measurements"]["range_rate"])

            pdirs["tx"]         = np.append(pdirs["tx"], dat["measurements"]["pointing_direction"]["tx"], axis=1)
            pdirs["rx"]         = np.append(pdirs["rx"], dat["measurements"]["pointing_direction"]["rx"], axis=1)
            
            if calculate_snr is True:
                snr             = np.append(snr, dat["measurements"]["snr"])
                rcs             = np.append(rcs, dat["measurements"]["rcs"])
                detection       = np.append(detection, dat["measurements"]["detection"])

                if doppler_spread_integrated_snr is True:
                    snr_inch        = np.append(snr_inch, dat["measurements"]["snr_inch"])

    # create final data structure
    data = dict()
    data["measurements"] = dict(
        t=t_ctrl,
        t_measurements=t_measurements,
        snr=snr,
        range=ranges,
        range_rx=range_rx,
        range_rate=range_rates,
        pointing_direction=pdirs,
        rcs=rcs,
        detection=detection)

    data["txi"] = txi
    data["rxi"] = rxi

    # add incoherent SNR measurements if provided
    if doppler_spread_integrated_snr is True:
        data["measurements"]["snr_inch"] = snr_inch

    return data


def get_max_snr_measurements(data, copy=True):
    ''' Returns the maximum SNR measurements for each time slice. 
    
    Measurements can be computed for multiple pointing directions (e.g. to allow
    simulation of digital beam steering). This function computes the maximum 
    SNR over each time slice and returns the corresponding measurements.

    .. seealso::
        Please refer to :attr:`Measurements.measure_states` for more information
        about the measurement data structure standards.

    Parameters
    ----------
    data : dict
        Measurement data structure containing multiple measurements per time slice.
    copy : bool
        If ``True``, the measurement data will be copied, allowing to keep the old 
        measurement data.

    Returns
    -------
    max_snr_data : dict
        New measurement data structure containing the maximum SNR measurements (range, range rate, ...).
    '''

    measurements = data["measurements"]
    if "snr" not in measurements.keys():
        raise Exception("No snr measurements found in the data")

    snr = measurements["snr"]
    t_measurements = measurements["t_measurements"]
    t = measurements["t"]

    inds = np.ndarray((len(measurements["t"]),), dtype=np.int32)

    clibsorts.get_max_snr_measurements.argtypes = [
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t.ndim, shape=t.shape),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t_measurements.ndim, shape=t_measurements.shape),
        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=snr.ndim, shape=snr.shape),
        np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=inds.ndim, shape=inds.shape),
        ctypes.c_int,
        ctypes.c_int,
    ]

    # call C library to get highest SNR measurement per time slice
    clibsorts.get_max_snr_measurements(
        t,
        t_measurements,
        snr,
        inds,
        ctypes.c_int(len(t_measurements)),
        ctypes.c_int(len(inds)),
    )

    # save data 
    if copy: 
        data = data.copy()

    pdirs = data["measurements"]["pointing_direction"]
    pdirs["tx"] = pdirs["tx"][:, inds]
    pdirs["rx"] = pdirs["rx"][:, inds]

    data["measurements"]["pointing_direction"]  = pdirs
    data["measurements"]["snr"]                 = data["measurements"]["snr"][inds]
    data["measurements"]["t_measurements"]      = data["measurements"]["t_measurements"][inds]
    data["measurements"]["range"]               = data["measurements"]["range"][inds]
    data["measurements"]["range_rate"]          = data["measurements"]["range_rate"][inds]
    data["measurements"]["rcs"]                 = data["measurements"]["rcs"][inds]
    data["measurements"]["detection"]           = data["measurements"]["detection"][inds]

    if "snr_inch" in data["measurements"].keys():
        data["measurements"]["snr_inch"] = data["measurements"]["snr_inch"][inds]

    return data