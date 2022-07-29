import numpy as np
import ctypes
import multiprocessing

import scipy.constants

from .. import signals
from ..passes import Pass, find_simultaneous_passes, equidistant_sampling
from ...common import interpolation 
from sorts import clibsorts

class Measurement_new(object):
    '''
    This class handles the computation/simulation of Radar measurements.
    Each radar system must be associated with a measurement unit to be able to simulate the observations resulting from given control sequence.
    '''
    def __init__(self, logger=None, profiler=None):
        '''
        '''
        pass

    def compute_measurement_jacobian(
        self, 
        radar_states, 
        space_object, 
        radar, 
        variables, 
        deltas, 
        **kwargs
        ):
        if self.logger is not None:
            self.logger.debug(f'Measurement:compute_measurement_jacobian: variables={variables}, deltas={deltas}')

        if self.profiler is not None:
            self.profiler.start('Measurement:compute_measurement_jacobian')
            self.profiler.start('Measurement:compute_measurement_jacobian:reference')

        data0 = self.calculate_observation(
            radar_states, 
            space_object, 
            radar, 
            **kwargs)
        
        if data0 is None:
            return None, None

        t       = data["t"]

        r       = np.ndarray((len(t),2), dtype=object)
        r_dot   = np.ndarray((len(t),2), dtype=object)
        keep    = np.ndarray((len(t),), dtype=object)
        dkeep   = np.ndarray((len(t),), dtype=object)

        J       = np.ndarray((len(t),), dtype=object)
        Jkeep   = np.ndarray((len(t),), dtype=object)

        # get reference values
        for period_id in range(len(t)):
            J = np.zeros([len(t[period_id])*2, len(variables)], dtype=np.float64)

            Jkeep[period_id] = np.full((len(t[period_id])*2,), False, dtype=np.bool)
            dkeep[period_id] = np.full(t[period_id].shape, False, dtype=np.bool)
            keep[period_id] = np.full(t[period_id].shape, False, dtype=np.bool)

            keep[period_id][data0['kept']] = True

            r[period_id] = np.ndarray((len(t[period_id]), 2), dtype=np.float64)
            r_dot[period_id] = np.ndarray((len(t[period_id]), 2), dtype=np.float64)

            r[period_id][keep[period_id], 0] = data0['range']
            r_dot[period_id][keep[period_id], 0] = data0['range_rate']


        if self.profiler is not None:
            self.profiler.stop('Measurement:compute_measurement_jacobian:reference')
            
        kwargs['snr_limit'] = False
        kwargs['calculate_snr'] = False

        for ind, var in enumerate(variables):
            if self.profiler is not None:
                self.profiler.start(f'Measurement:compute_measurement_jacobian:d_{var}')

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
            ddata = self.calculate_observation(
                radar_states, 
                dso, 
                radar, 
                **kwargs)

            # update values and compute jacobian
            for period_id in range(len(t)):
                dkeep[ddata['kept']] = True
                keep[period_id] = np.logical_and(keep[period_id], dkeep[period_id])

                r[period_id][dkeep[period_id], 1] = ddata['range']
                r_dot[period_id][dkeep[period_id], 1] = ddata['range_rate']

                dr = (r[period_id][:, 1] - r[period_id][:, 0])/deltas[ind]
                dv = (r_dot[period_id][:, 1] - r_dot[period_id][:, 0])/deltas[ind]

                J[period_id][:len(t), ind] = dr
                J[period_id][len(t):, ind] = dv

                for key in data0:
                    if key in ['kept']:
                        continue
                    elif isinstance(data0[key], np.ndarray):
                        data0[key] = data0[key][..., keep[period_id][data0['kept']]]
                    else:
                        data0[key] = [x for ind_, x in enumerate(data0[key]) if keep[period_id][data0['kept']][ind_]]
                data0['kept'] = np.argwhere(keep[period_id]).flatten()
                
                Jkeep[period_id][:len(t)]=keep[period_id]
                Jkeep[period_id][len(t):]=keep[period_id]

                J[period_id] = J[period_id][Jkeep[period_id], :]

            if self.profiler is not None:
                self.profiler.stop(f'Measurement:compute_measurement_jacobian:d_{var}')

        if self.profiler is not None:
            self.profiler.stop('Measurement:calculate_observation_jacobian')

        return data0, J


    def compute_space_object_measurements(
        self,
        radar_states, 
        space_object, 
        radar, 
        rx_indices=None,
        tx_indices=None,
        epoch=None, 
        calculate_snr=True, 
        doppler_spread_integrated_snr=False,
        interpolator=interpolation.Legendre8, 
        max_dpos=100e3,
        snr_limit=True, 
        save_states=False, 
        logger=None,
        profiler=None,
        parallelization=True,
        n_processes=16,
        ):
        if radar_states.pdirs is None: # no pointing directions provided, so the time array will be the same as the time slice array
            if logger is not None: 
                logger.debug(f'Measurements:compute_space_object_measurements: No pointing directions in current radar states, ignoring pointing directions...')
            t_dirs = lambda period_id : radar_states.t[period_id]
        else:
            t_dirs = lambda period_id : radar_states.pdirs[period_id]["t"]
        
        if profiler is not None: 
            profiler.start('Measurements:compute_space_object_measurements')

        # Checking input station indices
        tx_indices, rx_indices = self.__get_station_indices(tx_indices, rx_indices, radar)
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

        for txi in range(len(tx_indices)):
            for rxi in range(len(rx_indices)):
                # get passes
                if profiler is not None: profiler.start('Measurements:compute_space_object_measurements:Initialization:find_passes')
                passes = find_simultaneous_passes(t_states, states, [radar.tx[tx_indices[txi]], radar.rx[rx_indices[rxi]]])
                if profiler is not None: profiler.stop('Measurements:compute_space_object_measurements:Initialization:find_passes')

                if logger is not None:
                    logger.info(f"found {len(passes)} passes for stations [tx:{tx_indices[txi]}, rx:{rx_indices[rxi]}]")

                if len(passes) > 0:
                    for pass_id, pass_ in enumerate(passes):
                        if logger is not None:
                            logger.info(f"Computing measurement of pass {pass_id} for stations [tx:{tx_indices[txi]}, rx:{rx_indices[rxi]}]")

                        pass_.station_id = [tx_indices[txi], rx_indices[rxi]]
                        data["pass_data"].append(self.compute_pass_measurements(
                            pass_,
                            radar_states, 
                            space_object, 
                            radar, 
                            epoch=epoch, 
                            calculate_snr=calculate_snr, 
                            doppler_spread_integrated_snr=doppler_spread_integrated_snr,
                            interpolator=interpolator, 
                            max_dpos=max_dpos,
                            snr_limit=snr_limit, 
                            save_states=save_states, 
                            logger=logger,
                            profiler=profiler,
                            parallelization=parallelization,
                            n_processes=n_processes,
                        ))

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
        radar, 
        epoch=None, 
        calculate_snr=True, 
        doppler_spread_integrated_snr=False,
        interpolator=interpolation.Legendre8, 
        max_dpos=100e3,
        snr_limit=True, 
        save_states=False, 
        logger=None,
        profiler=None,
        parallelization=True,
        n_processes=16,
        ):
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
            if txrx_pass.enu is not None:
                tx_enu = txrx_pass.enu[0] # tx_enus/rx_enus
                states = radar.tx[txi].to_ecef(tx_enu)
                t_states = txrx_pass.t
            else: 
                # propagate to get low-res states
                t_start = t_dirs(0)[0]
                t_end = t_dirs(n_periods-1)[-1]
                t_states, states = self.get_states_low_resolution(
                    t_start, 
                    t_end, 
                    space_object, 
                    epoch=epoch, 
                    max_dpos=max_dpos,
                    logger=logger,
                    profiler=profiler,
                )

            # initializes the states interpolator
            state_interpolator = interpolator(states, t_states)
        else:
            state_interpolator = interpolator

        t_sampling_states = np.ndarray(0, dtype=float)
        space_object_states = np.ndarray((6, 0), dtype=float)
        data = []

        # compute measurements for each period id
        for period_id in range(n_periods):
            t_tmp = t_dirs(period_id)
            if t_tmp[-1] > txrx_pass.start():
                # propagate states over subperiod
                tmp_states = state_interpolator.get_state(t_tmp)

                b_sup   = min(txrx_pass.end(), t_tmp[-1])
                b_inf   = max(txrx_pass.start(), t_tmp[0])
                bounds  = np.array([b_inf, b_sup])

                # clean radar states
                radar_properties = dict()
                for station_type in ("tx", "rx"):
                    radar_properties[station_type] = radar_states.property_controls[period_id][station_type]

                if pdirs(period_id) is None:
                    pdirs_tx = None
                    pdirs_rx = None
                else:
                    pdirs_tx = pdirs(period_id)["tx"][txi, 0]
                    pdirs_rx = pdirs(period_id)["rx"][rxi, txi]

                data.append(self.parallel_states_measurements(
                    radar_states.t[period_id],
                    radar_states.t_slice[period_id],
                    t_tmp, 
                    pdirs_tx, 
                    pdirs_rx, 
                    space_object,
                    tmp_states, 
                    radar, 
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
                ))

                # save states
                if save_states is True:
                    space_object_states = np.append(space_object_states, tmp_states[:, np.logical_and(t_tmp>=bounds[0], t_tmp<=bounds[1])], axis=1)

                if t_tmp[-1] > txrx_pass.end():
                    break

        data_final = self.recover_data(txi, rxi, data, calculate_snr, doppler_spread_integrated_snr)
        if save_states is True:
            data_final["states"] = space_object_states

        return data_final


    def parallel_states_measurements(
        self,
        t,
        t_slice,
        t_dirs, 
        pdirs_tx, 
        pdirs_rx, 
        space_object,
        space_object_states, 
        radar, 
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
        parallelization=True,
        n_processes=16,
        ):
        ''' Simulates an observation over a single control period. '''

        # get object properties
        diameter                    = space_object.d
        spin_period                 = space_object.parameters.get('spin_period', 0.0)
        radar_albedo                = space_object.parameters.get('radar_albedo', 1.0)

        # bounds is used instead of the mask to synchronize the controls and pdirs time arrays
        if bounds is None:
            start_index     = 0
            end_index       = len(t) - 1
        else:
            bounds_id       = self.get_bounds(t, bounds)
            start_index     = bounds_id[0]
            end_index       = bounds_id[1]

        if parallelization is True:
            def process_function(pid, sub_bounds, return_dict):
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
                )

            # compute parallelization sub array properties (for control arrays)
            n_time_points = end_index - start_index + 1
            n_max_points_per_period = int(np.ceil(n_time_points/n_processes))
            if n_max_points_per_period == 0: # if there are less points than processes, reduce number of processes to match number of points
                n_processes = n_time_points
                n_max_points_per_period = 1
                n_points_last_period = 1
            else:
                n_points_last_period = n_time_points%((n_processes-1)*n_max_points_per_period)

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

            # retreive computation results  
            if parallelization is True:
                # wait for each process to be finished
                for pid, process in enumerate(process_subgroup):
                    process.join()
                    data[pid] = return_dict[pid]

            data_final = self.recover_data(txi, rxi, data, calculate_snr, doppler_spread_integrated_snr)
        else:
            # compute measurements without paralellization
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
            )

        if save_states:
            data_final['states'] = space_object_states

        return data_final


    def get_states_low_resolution(
        self,
        t_start, 
        t_end, 
        space_object, 
        epoch=None, 
        max_dpos=100e3,
        logger=None,
        profiler=None,
    ):
        if epoch is not None:
            dt_epoch = (epoch - space_object.epoch).sec
        else:
            dt_epoch = 0

        if profiler is not None: 
            profiler.start('Measurements:Measure:Initialization:create_sampling_time_array')

        # create state sampling time array
        t_states = equidistant_sampling(orbit=space_object.state, start_t=t_start, end_t=t_end, max_dpos=max_dpos) + dt_epoch
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
        profiler=None,
        logger=None,
    ):
        # if the state mask is not provided, define new state mask for measurement computations
        if bounds is None:
            bounds = [t[0], t[-1]]
        bound_indices_pdirs = self.get_bounds(t_dirs, bounds)
        bound_indices_ctrl  = self.get_bounds(t, bounds)

        n_dirs              = bound_indices_pdirs[1] - bound_indices_pdirs[0] + 1
        n_time_slices       = bound_indices_ctrl[1] - bound_indices_ctrl[0] + 1

        # compute range and range rates
        if profiler is not None: profiler.start('Measurements:Measure:enus,range,range_rate')
        enu_tx_to_so, enu_rx_to_so, ranges_tx, ranges_rx, range_rates_tx, range_rates_rx = self.__compute_ranges(object_states[:, bound_indices_pdirs[0]:bound_indices_pdirs[-1]+1], radar, txi, rxi)
        if profiler is not None: profiler.stop('Measurements:Measure:enus,range,range_rate')

        ranges            = ranges_tx + ranges_rx
        range_rates       = range_rates_tx + range_rates_rx

        # compute SNR
        if calculate_snr:
            if profiler is not None: profiler.start('Measurements:Measure:rcs')
            rcs = signals.hard_target_rcs(wavelength=radar_properties["tx"]["wavelength"][txi], diameter=diameter)       
            if profiler is not None: profiler.stop('Measurements:Measure:rcs')

             # compute beam gains
            if profiler is not None: profiler.start('Measurements:Measure:beam_gain')
            tx_gain, rx_gain = self.compute_gain(
                t, 
                t_dirs, 
                pdirs_tx, 
                pdirs_rx, 
                radar, 
                radar_properties, 
                controlled_properties, 
                enu_tx_to_so, 
                enu_rx_to_so, 
                txi, 
                rxi, 
                profiler=profiler,
                bounds_pdirs_id=bound_indices_pdirs, 
                bounds_ctrl_id=bound_indices_ctrl,
            )
            if profiler is not None: profiler.stop('Measurements:Measure:beam_gain')

            if profiler is not None: 
                profiler.start('Measurements:Measure:snr')
            
            # snr data arrays
            snr             = np.ndarray((n_dirs,), dtype=np.float64)
            snr_inch        = np.ndarray((n_dirs,), dtype=np.float64)
            detection       = np.ones((n_dirs,), dtype=np.int32)

            clibsorts.compute_measurement_snr.argtypes = [
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t.ndim, shape=t.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t_dirs.ndim, shape=t_dirs.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t_slice.ndim, shape=t_slice.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["pulse_length"][txi].ndim, shape=radar_properties["tx"]["pulse_length"][txi].shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["ipp"][txi].ndim, shape=radar_properties["tx"]["ipp"][txi].shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=tx_gain.ndim, shape=tx_gain.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=rx_gain.ndim, shape=rx_gain.shape),
                np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["wavelength"][txi].ndim, shape=radar_properties["tx"]["wavelength"][txi].shape),
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
                ctypes.c_int,
                ctypes.c_int,
                ]

            clibsorts.compute_measurement_snr(
                t,
                t_dirs,
                t_slice,
                radar_properties["tx"]["pulse_length"][txi],
                radar_properties["tx"]["ipp"][txi],
                tx_gain,
                rx_gain,
                radar_properties["tx"]["wavelength"][txi],
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
                ctypes.c_int(n_time_slices),
                ctypes.c_int(n_dirs),
                ctypes.c_int(bound_indices_ctrl[0]),
                ctypes.c_int(bound_indices_pdirs[0]),
            )

            if profiler is not None: 
                profiler.stop('Measurements:Measure:snr')
        else:
            snr             = None
            snr_inch        = None
            rcs             = None
            detection       = None

        # remove pointing directions outside of pass
        if pdirs_tx is not None:
            pdirs_tx[:, bound_indices_pdirs[0]:bound_indices_pdirs[1]+1]
        if pdirs_rx is not None:
            pdirs_rx[:, bound_indices_pdirs[0]:bound_indices_pdirs[1]+1]

        pointing_direction = dict()
        pointing_direction["tx"] = pdirs_tx
        pointing_direction["rx"] = pdirs_rx

        #exit()
        data = dict()
        data["measurements"] = dict(
            t=t[bound_indices_ctrl[0]:bound_indices_ctrl[1]+1],
            t_measurements=t_dirs[bound_indices_pdirs[0]:bound_indices_pdirs[1]+1],
            snr=snr,
            range=ranges,
            range_rate=range_rates,
            pointing_direction=pointing_direction,
            rcs=rcs,
            txi=txi,
            rxi=rxi,
            detection=detection)

        if doppler_spread_integrated_snr:
            data['snr_inch'] = snr_inch

        return data


    def stop_condition(self, t, radar, radar_states):
        '''
        Measurement abort/stop condition (i.e. stop time, ...)
        '''
        return False


    def observable_filter(self, t, radar, radar_states, tx_enu, rx_enu, txi, rxi):
        '''
            Determines if the object was observable or not by the radar system
        '''
        return True


    def __get_station_indices(self, tx_indices, rx_indices, radar):
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


    def __compute_ranges(self, states, radar, txi, rxi):
        # Transmitters
        enu_tx_to_so     = radar.tx[txi].enu(states)
        ranges_tx        = Pass.calculate_range(enu_tx_to_so)
        range_rates_tx   = Pass.calculate_range_rate(enu_tx_to_so)

        # Receivers
        enu_rx_to_so     = radar.rx[rxi].enu(states)
        ranges_rx        = Pass.calculate_range(enu_rx_to_so)
        range_rates_rx   = Pass.calculate_range_rate(enu_rx_to_so)

        return enu_tx_to_so, enu_rx_to_so, ranges_tx, ranges_rx, range_rates_tx, range_rates_rx


    def get_beam_gain(self, beam, enu):
        '''Given the input beam configured by the controller and the local (for that beam) coordinates of the observed object, get the correct gain and wavelength used for SNR calculation. 

        The default is a maximum calculation based on pointing, this is used for e.g. RX digital beam-forming.
        '''
        if len(beam.pointing.shape) > 1:
            g = np.max([
                beam.gain(enu, ind={'pointing': pi})
                for pi in range(beam.pointing.shape[1])
            ])
        else:
            g = beam.gain(enu)

        return g


    def compute_gain(
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
        bounds_pdirs_id=None, 
        bounds_ctrl_id=None, 
        profiler=None
        ):
        '''
        Given the input beam configured by the controller and the local (for that beam) coordinates of the observed object, get the correct gain and wavelength used for SNR calculation. 

        The default is a maximum calculation based on pointing, this is used for e.g. RX digital beam-forming.
        '''
        # compensates for eventual desync between time and control arrays
        if bounds_pdirs_id is None:
            bounds_pdirs_id = np.array([0, len(t_dirs)-1], dtype=np.int32)

        if bounds_ctrl_id is None:
            bounds_ctrl_id = np.array([0, len(t)-1], dtype=np.int32)

        N = len(t[bounds_ctrl_id[0]:bounds_ctrl_id[-1]+1])
        N_dirs = len(t_dirs[bounds_pdirs_id[0]:bounds_pdirs_id[-1]+1])
        
        gain_tx = np.zeros((N_dirs,), dtype=np.float64)
        gain_rx = np.zeros((N_dirs,), dtype=np.float64)

        def tx_gain(t_dir, ti):
            nonlocal property_controls, controlled_properties
            station = radar.tx[txi]

            if profiler is not None: 
                profiler.start('Measurements:compute_gain:set_properties')

            # set tx properties
            if pdirs_tx is not None:
                station.point_ecef(pdirs_tx[:, t_dir])
            for property_name in controlled_properties["tx"][txi]:
                if hasattr(station, property_name):
                    exec("radar.tx[txi]." + property_name + " = property_controls['tx'][property_name][txi][ti]")

            if profiler is not None: 
                profiler.stop('Measurements:compute_gain:set_properties')
                profiler.start('Measurements:compute_gain:gain')

            gain_tx[t_dir - bounds_pdirs_id[0]] = self.get_beam_gain(station.beam, tx_enus[0:3, t_dir - bounds_pdirs_id[0]])

            if profiler is not None: 
                profiler.stop('Measurements:compute_gain:gain')


        def rx_gain(t_dir, ti):
            nonlocal property_controls, controlled_properties
            station = radar.rx[rxi]

            if profiler is not None: 
                profiler.start('Measurements:compute_gain:set_properties')

            # set beam properties
            if pdirs_rx is not None:
                station.point_ecef(pdirs_rx[:, t_dir])
            for property_name in controlled_properties["rx"][rxi]:
                if hasattr(station, property_name):
                    exec("radar.rx[rxi]." + property_name + " = property_controls['rx'][property_name][rxi][ti]")

            if profiler is not None: 
                profiler.stop('Measurements:compute_gain:set_properties')
                profiler.start('Measurements:compute_gain:gain')

            gain_rx[t_dir - bounds_pdirs_id[0]] = self.get_beam_gain(station.beam, rx_enus[0:3, t_dir - bounds_pdirs_id[0]])

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
            ctypes.c_int(bounds_ctrl_id[0]),
            ctypes.c_int(bounds_pdirs_id[0]),
            compute_antenna_gain_tx_c,
            compute_antenna_gain_rx_c,
            )

        return gain_tx, gain_rx


    def recover_data(self, txi, rxi, datas, calculate_snr, doppler_spread_integrated_snr):
        ''' Fuses multiple computation results into a single structure. '''

        # recover and reshape data to pass
        t_ctrl              = np.ndarray(0, dtype=float)
        t_measurements      = np.ndarray(0, dtype=float)
        ranges              = np.ndarray(0, dtype=float)
        range_rates         = np.ndarray(0, dtype=float)

        pdirs = dict()
        pdirs["tx"]         = np.ndarray((3, 0), dtype=float)
        pdirs["rx"]         = np.ndarray((3, 0), dtype=float)

        if calculate_snr is True:
            snr             = np.ndarray(0, dtype=float)
            snr_inch        = np.ndarray(0, dtype=float)
            rcs             = np.ndarray(0, dtype=float)
            detection       = np.ndarray(0, dtype=np.int32)
        else:
            snr             = None
            snr_inch        = None
            rcs             = None
            detection       = None

        for i, dat in enumerate(datas):
            if dat is not None:
                t_ctrl              = np.append(t_ctrl, dat["measurements"]["t"])
                t_measurements      = np.append(t_measurements, dat["measurements"]["t_measurements"])
                ranges              = np.append(ranges, dat["measurements"]["range"])
                range_rates         = np.append(range_rates, dat["measurements"]["range_rate"])

                pdirs["tx"]         = np.append(pdirs["tx"], dat["measurements"]["pointing_direction"]["tx"], axis=1)
                pdirs["rx"]         = np.append(pdirs["rx"], dat["measurements"]["pointing_direction"]["rx"], axis=1)
                
                if calculate_snr is True:
                    snr             = np.append(snr, dat["measurements"]["snr"])
                    rcs             = np.append(rcs, dat["measurements"]["rcs"])
                    detection       = np.append(detection, dat["measurements"]["detection"])

                    if doppler_spread_integrated_snr is True:
                        snr_inch        = np.append(snr_inch, dat["measurements"]["snr_inch"])

        data = dict()
        data["measurements"] = dict(
            t=t_ctrl,
            t_measurements=t_measurements,
            snr=snr,
            range=ranges,
            range_rate=range_rates,
            pointing_direction=pdirs,
            rcs=rcs,
            txi=txi,
            rxi=rxi,
            detection=detection)
        
        if doppler_spread_integrated_snr is True:
            data["measurements"]["snr_inch"] = snr_inch

        return data



    def get_bounds(self, t, bounds):
        ''' Gets the indices of the bounds of an ordered time array. ''' 
        N = len(t)

        if bounds is None:
            return np.array([0, N-1])
        else:
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



def get_max_snr_measurements(data, copy=True):
    ''' If multiple measurements are present during a given time slice, only the measurement with the maximum SNR will be kept. '''

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