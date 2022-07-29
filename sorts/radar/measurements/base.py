import numpy as np
import ctypes
from multiprocessing import Process, Lock, Manager

import scipy.constants

from .. import signals
from ..passes import Pass, find_simultaneous_passes, equidistant_sampling

from sorts import clibsorts

from ...common import interpolation 

class Measurement(object):
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

        if self.profiler is not None:
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

    def measure(
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
        if radar_states.pdirs is None:
            if logger is not None: 
                logger.debug(f'Measurements:Measure: No pointing directions in current radar states, ignoring pointing directions...')
            
            # no pointing directions provided, so the time array will be the same as the time slice array
            t_dirs = lambda period_id : radar_states.t[period_id]
            pdirs = lambda period_id : None
        else:
            t_dirs = lambda period_id : radar_states.pdirs[period_id]["t"]
            pdirs = lambda period_id : radar_states.pdirs[period_id]
        
        if profiler is not None: 
            profiler.start('Measurements:Measure')

        # Checking input station indices
        tx_indices, rx_indices = self.__get_station_indices(tx_indices, rx_indices, radar)
        if logger is not None: 
            logger.debug(f'Measurements:Measure: stations tx={tx_indices}, rx={rx_indices}')

        if not issubclass(interpolator, interpolation.Interpolator):
            raise TypeError(f"interpolator must be an instance of {interpolation.Interpolator}.")


        if profiler is not None: 
            profiler.start('Measurements:Measure:Initialization')

        n_periods                   = len(radar_states.t)
        space_object_states         = np.ndarray((n_periods,), dtype=object)
        pass_mask                   = np.ndarray((n_periods,), dtype=object)

        diameter                    = space_object.d
        spin_period                 = space_object.parameters.get('spin_period', 0.0)
        radar_albedo                = space_object.parameters.get('radar_albedo', 1.0)


        # Initializing computation results
        data = dict()
        data["measurements"] = np.ndarray((n_periods,), dtype=object)

        # t is always in scheduler relative time
        # t_samp is in space object relative time if there is a scheduler epoch, otherwise it is assumed that the epoch are the same
        # if there is an interpolator it is assumed that interpolation is done in space object relative time
        if profiler is not None: 
            profiler.start('Measurements:Measure:Initialization:create_sampling_time_array')

        if epoch is not None:
            dt_epoch = (epoch - space_object.epoch).sec
        else:
            dt_epoch = 0

        # create state sampling time array
        t_states = equidistant_sampling(orbit=space_object.state, start_t=t_dirs(0)[0], end_t=t_dirs(n_periods-1)[-1], max_dpos=max_dpos) + dt_epoch
        t_states = np.append(t_states, t_states[-1] + (t_states[-1] - t_states[-2])) # add an extra point to the propagation array because last time point is not attained
        if profiler is not None: 
            profiler.stop('Measurements:Measure:Initialization:create_sampling_time_array')
            profiler.start('Measurements:Measure:Initialization:get_object_states')

        # propagate space object states
        states = space_object.get_state(t_states)
        state_interpolator = interpolator(states, t_states)

        if profiler is not None: 
            profiler.stop('Measurements:Measure:Initialization:get_object_states')

        for period_id in range(n_periods):
            t_sampling_states = t_dirs(period_id) + dt_epoch
            space_object_states[period_id] = state_interpolator.get_state(t_sampling_states)

            # the t_sampling_states array is always within the t_states array interval, but it can be integrally in between two points, so we need
            # to get the indices of t_states (low-res) which contain the whole t_sampling_states interval
            i_start = np.where(t_states <= t_sampling_states[0])[0][-1]
            i_end = np.where(t_states >= t_sampling_states[-1])[0][0]
            
            pass_mask[period_id] = np.zeros((len(tx_indices), len(rx_indices), len(t_sampling_states),), dtype=np.int32)
            for txi in range(len(tx_indices)):
                for rxi in range(len(rx_indices)):
                    if profiler is not None: 
                        profiler.start('Measurements:Measure:Initialization:find_passes')
                    
                    passes = find_simultaneous_passes(t_states[i_start:i_end], states[:,i_start:i_end], [radar.tx[tx_indices[txi]], radar.rx[rx_indices[rxi]]])
                    
                    if profiler is not None: 
                        profiler.stop('Measurements:Measure:Initialization:find_passes')

                    if len(passes) > 0:
                        for _pass_id, _pass in enumerate(passes):
                            t_pass_start_id_lowres = _pass.inds[0] + i_start
                            t_pass_end_id_lowres = _pass.inds[-1] + i_start
                            if i_start > 0:                   
                                t_pass_start_id_lowres -= 1  
                            if t_pass_end_id_lowres < len(t_states)-1:
                                t_pass_end_id_lowres+= 1


                            pass_mask_tmp = np.where(np.logical_and(t_sampling_states >= t_states[t_pass_start_id_lowres], t_sampling_states <= t_states[t_pass_end_id_lowres]))[0]
                            
                            if len(pass_mask_tmp) > 0:
                                t_pass_start_id = pass_mask_tmp[0]
                                t_pass_end_id = pass_mask_tmp[-1]+1
                                pass_mask[period_id][txi, rxi][t_pass_start_id:t_pass_end_id] = 1
        
        if save_states:
            data['states'] = space_object_states
            data['pass_mask'] = pass_mask.copy()

        if profiler is not None: 
            profiler.stop('Measurements:Measure:Initialization')

        def process_function(pid, period_id, return_dict):
            # clean radar states
            radar_properties = dict()
            for station_type in ("tx", "rx"):
                radar_properties[station_type] = radar_states.property_controls[period_id][station_type]

            tmp_data = None

            if pass_mask[period_id].any() == 1:
                return_dict[pid] = self.measure_states(
                    radar_states.t[period_id], 
                    radar_states.t_slice[period_id],
                    t_dirs(period_id), 
                    space_object_states[period_id], 
                    radar, 
                    pdirs(period_id), 
                    radar_properties, 
                    radar_states.controlled_properties, 
                    diameter,
                    spin_period,
                    radar_albedo,
                    tx_indices=tx_indices, 
                    rx_indices=rx_indices,
                    state_mask=pass_mask[period_id],
                    calculate_snr=calculate_snr, 
                    doppler_spread_integrated_snr=doppler_spread_integrated_snr,
                    snr_limit=snr_limit, 
                    profiler=profiler,
                    logger=logger,
                )
            else:
                 return_dict[pid] = None

        manager = Manager()
        return_dict = manager.dict()

        # paralellize computations over each period ids if option enabled
        if parallelization is True:
            for process_subgroup_id in range(int(n_periods/n_processes) + 1):
                if int(n_periods - process_subgroup_id*n_processes) >= n_processes:
                    n_process_in_subgroup = n_processes
                else:
                    n_process_in_subgroup = int(n_periods - process_subgroup_id*n_processes)

                process_subgroup = []

                # initializes each process and associate them to an object in the list of targets to follow
                for i in range(n_process_in_subgroup):
                    period_id = process_subgroup_id * n_processes + i # get the period id
                    process = Process(target=process_function, args=(i, period_id, return_dict)) # create new process

                    if logger is not None: 
                        logger.info(f"TrackingScheduler:generate_schedule -> (process pid {i}) creating subprocess id {period_id}") 

                    process_subgroup.append(process)
                    process.start()

                # wait for each process to be finished
                for pid, process in enumerate(process_subgroup):
                    process.join()

                for pid, process in enumerate(process_subgroup):
                    period_id = process_subgroup_id * n_processes + pid # get the period id
                    data["measurements"][period_id] = return_dict[pid]
        else:
            # if parallelization is disabled
            for period_id in range(n_periods):
                process_function(period_id, period_id, None, data["measurements"])

        if profiler is not None: 
            profiler.stop('Measurements:Measure')

        if logger is not None:
            logger.debug(f'Measure:completed')

        return data

    def measure_states(
        self,
        t, 
        t_slice,
        t_dirs, 
        object_states, 
        radar, 
        radar_pdirs, 
        radar_properties, 
        controlled_properties,
        diameter,
        spin_period,
        radar_albedo,
        tx_indices=None, 
        rx_indices=None,
        state_mask=None,
        calculate_snr=True, 
        doppler_spread_integrated_snr=False,
        snr_limit=False, 
        profiler=None,
        logger=None,
    ):
        n_dirs                      = len(t_dirs)
        n_time_slices               = len(t)

        # if the state mask is not provided, define new state mask for measurement computations
        if state_mask is None:
            state_mask = np.full((len(tx_indices), len(rx_indices), n_dirs,), True, dtype=bool)
        else:
            if np.shape(state_mask) != (len(tx_indices), len(rx_indices), n_dirs,):
                raise ValueError(f"state_mask shape must be {(len(tx_indices), len(rx_indices) ,n_dirs,)}, not {np.shape(state_mask)}.")

        if tx_indices is None:
            tx_indices = np.arange(0, len(radar.tx))
        if rx_indices is None:
            rx_indices = np.arange(0, len(radar.rx))

        ranges = np.ndarray((len(tx_indices), len(rx_indices), n_dirs))
        range_rates = np.ndarray((len(tx_indices), len(rx_indices), n_dirs))

        # define output results
        if calculate_snr is True:
            snr             = np.ndarray((len(tx_indices), len(rx_indices),), dtype=object)
            snr_inch        = np.ndarray((len(tx_indices), len(rx_indices),), dtype=object)
            rcs             = np.ndarray((len(tx_indices), n_time_slices,), dtype=np.float64)
            detection       = np.ndarray((len(tx_indices), len(rx_indices),), dtype=object)
        else:
            snr             = None
            snr_inch        = None
            rcs             = None
            detection       = None

        # compute range and range rates
        if profiler is not None: 
            profiler.start('Measurements:Measure:enus,range,range_rate')
       
        enu_tx_to_so, enu_rx_to_so, ranges_tx, ranges_rx, range_rates_tx, range_rates_rx = self.__compute_ranges(object_states, radar, tx_indices, rx_indices)

        if profiler is not None: 
            profiler.stop('Measurements:Measure:enus,range,range_rate')

        # compute beam gains
        if profiler is not None: 
            profiler.start('Measurements:Measure:beam_gain')
        
        tx_gain, rx_gain = self.compute_gain(t, t_dirs, radar_pdirs, radar, radar_properties, controlled_properties, enu_tx_to_so, enu_rx_to_so, state_mask, tx_indices, rx_indices, profiler=profiler)

        if profiler is not None: 
            profiler.stop('Measurements:Measure:beam_gain')

        # compute SNR
        if calculate_snr:
            for txi in tx_indices:
                if profiler is not None: 
                    profiler.start('Measurements:Measure:rcs')
                
                rcs[txi] = signals.hard_target_rcs(wavelength=radar_properties["tx"]["wavelength"][txi], diameter=diameter)
                
                if profiler is not None: 
                    profiler.stop('Measurements:Measure:rcs')

                for rxi in rx_indices:
                    snr[txi, rxi]               = np.ndarray((n_dirs,), dtype=np.float64)                    
                    snr_inch[txi, rxi]          = np.ndarray((n_dirs,), dtype=np.float64)    
                    ranges[txi, rxi]            = ranges_tx[txi] + ranges_rx[rxi]
                    range_rates[txi, rxi]       = range_rates_tx[txi] + range_rates_rx[rxi]

                    if profiler is not None: 
                        profiler.start('Measurements:Measure:snr')
                    
                    clibsorts.compute_measurement_snr.argtypes = [
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t.ndim, shape=t.shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t_dirs.ndim, shape=t_dirs.shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t_slice.ndim, shape=t_slice.shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["pulse_length"][txi].ndim, shape=radar_properties["tx"]["pulse_length"][txi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["ipp"][txi].ndim, shape=radar_properties["tx"]["ipp"][txi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=tx_gain[txi].ndim, shape=tx_gain[txi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=rx_gain[rxi].ndim, shape=rx_gain[rxi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["wavelength"][txi].ndim, shape=radar_properties["tx"]["wavelength"][txi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["power"][txi].ndim, shape=radar_properties["tx"]["power"][txi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=ranges_tx[txi].ndim, shape=ranges_tx[txi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=ranges_rx[rxi].ndim, shape=ranges_rx[rxi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["duty_cycle"][txi].ndim, shape=radar_properties["tx"]["duty_cycle"][txi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_properties["tx"]["coh_int_bandwidth"][txi].ndim, shape=radar_properties["tx"]["coh_int_bandwidth"][txi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=snr[txi, rxi].ndim, shape=snr[txi, rxi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=snr_inch[txi, rxi].ndim, shape=snr_inch[txi, rxi].shape),
                        np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=state_mask[txi, rxi].ndim, shape=state_mask[txi, rxi].shape),
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
                        ctypes.c_double,
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
                        tx_gain[txi],
                        rx_gain[rxi],
                        radar_properties["tx"]["wavelength"][txi],
                        radar_properties["tx"]["power"][txi],
                        ranges_tx[txi],
                        ranges_rx[rxi],
                        radar_properties["tx"]["duty_cycle"][txi],
                        radar_properties["tx"]["coh_int_bandwidth"][txi],
                        snr[txi, rxi],
                        snr_inch[txi, rxi],
                        state_mask[txi, rxi],
                        ctypes.c_double(radar.rx[rxi].noise_temperature),
                        ctypes.c_double(diameter),
                        ctypes.c_double(radar_albedo),
                        ctypes.c_double(spin_period),
                        ctypes.c_double(radar.min_SNRdb),
                        ctypes.c_int(doppler_spread_integrated_snr),
                        ctypes.c_int(snr_limit),
                        ctypes.c_int(n_dirs))

                    if profiler is not None: 
                        profiler.stop('Measurements:Measure:snr')

        data = dict(
            t=t,
            t_dirs=t_dirs,
            snr=snr,
            range=ranges,
            range_rate=range_rates,
            pointing_direction=radar_pdirs,
            rcs=rcs,
            tx_indices=tx_indices,
            rx_indices=rx_indices,
            detection=state_mask)

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

    def __compute_ranges(self, states, radar, tx_indices, rx_indices):
        N_points = len(states[0])

        # Transmitters
        enu_tx_to_so = np.empty((len(tx_indices), 6, N_points)) # vector from tx to space object
        ranges_tx = np.empty((len(tx_indices), N_points))
        range_rates_tx = np.empty((len(tx_indices), N_points))

        for txi in tx_indices:
            enu_tx_to_so[txi] = radar.tx[txi].enu(states)
            ranges_tx[txi] = Pass.calculate_range(enu_tx_to_so[txi])
            range_rates_tx[txi] = Pass.calculate_range_rate(enu_tx_to_so[txi])

        # Receivers
        enu_rx_to_so = np.empty((len(rx_indices), 6, N_points)) # vector from rx to space object
        ranges_rx = np.empty((len(rx_indices), N_points))
        range_rates_rx = np.empty((len(rx_indices), N_points))

        for rxi in rx_indices:
            enu_rx_to_so[rxi] = radar.rx[rxi].enu(states)
            ranges_rx[rxi] = Pass.calculate_range(enu_rx_to_so[rxi])
            range_rates_rx[rxi] = Pass.calculate_range_rate(enu_rx_to_so[rxi])

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
        pdirs, 
        radar, 
        property_controls, 
        controlled_properties, 
        tx_enus, 
        rx_enus, 
        pass_mask, 
        tx_indices, 
        rx_indices, 
        profiler=None
        ):
        '''
        Given the input beam configured by the controller and the local (for that beam) coordinates of the observed object, get the correct gain and wavelength used for SNR calculation. 

        The default is a maximum calculation based on pointing, this is used for e.g. RX digital beam-forming.
        '''
        N = len(t)
        N_dirs = len(t_dirs)
        
        gain_tx = np.ndarray((len(tx_indices), N_dirs,), dtype=np.float64)
        gain_rx = np.ndarray((len(rx_indices), len(tx_indices), N_dirs,), dtype=np.float64)

        def tx_gain(txi, ti, t_dir):
            nonlocal property_controls, controlled_properties
            tx_id = tx_indices[txi]
            station = radar.tx[tx_id]

            if profiler is not None: 
                profiler.start('Measurements:compute_gain:set_properties')

            # set tx properties
            if pdirs is not None:
                station.point_ecef(pdirs["tx"][txi, 0, :, t_dir])
            for property_name in controlled_properties["tx"][tx_id]:
                if hasattr(station, property_name):
                    exec("radar.tx[tx_id]." + property_name + " = property_controls['tx'][property_name][txi][ti]")

            if profiler is not None: 
                profiler.stop('Measurements:compute_gain:set_properties')
                profiler.start('Measurements:compute_gain:gain')

            gain_tx[txi, t_dir] = self.get_beam_gain(station.beam, tx_enus[txi, 0:3, t_dir])

            if profiler is not None: 
                profiler.stop('Measurements:compute_gain:gain')


        def rx_gain(rxi, txi, ti, t_dir):
            nonlocal property_controls, controlled_properties
            rx_id = rx_indices[rxi]
            station = radar.rx[rx_id]

            if profiler is not None: 
                profiler.start('Measurements:compute_gain:set_properties')

            # set beam properties
            if pdirs is not None:
                station.point_ecef(pdirs["rx"][rxi, txi, :, t_dir])
            for property_name in controlled_properties["rx"][rx_id]:
                if hasattr(station, property_name):
                    exec("radar.rx[rx_id]." + property_name + " = property_controls['rx'][property_name][rxi][ti]")

            if profiler is not None: 
                profiler.stop('Measurements:compute_gain:set_properties')
                profiler.start('Measurements:compute_gain:gain')

            gain_rx[rxi, txi, t_dir] = self.get_beam_gain(station.beam, rx_enus[rxi, 0:3, t_dir])

            if profiler is not None: 
                profiler.stop('Measurements:compute_gain:gain')


        # define callback functions 
        COMPUTE_GAIN_TX = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        COMPUTE_GAIN_RX = ctypes.CFUNCTYPE(None, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        compute_antenna_gain_tx_c = COMPUTE_GAIN_TX(tx_gain)
        compute_antenna_gain_rx_c = COMPUTE_GAIN_RX(rx_gain)


        clibsorts.compute_gain.argtypes = [
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t.ndim, shape=t.shape),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=t_dirs.ndim, shape=t_dirs.shape),
            np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=pass_mask.ndim, shape=pass_mask.shape),
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
            pass_mask.astype(np.int32),
            ctypes.c_int(N),
            ctypes.c_int(N_dirs),
            ctypes.c_int(len(tx_indices)),
            ctypes.c_int(len(rx_indices)),
            compute_antenna_gain_tx_c,
            compute_antenna_gain_rx_c,
        )

        return gain_tx, gain_rx