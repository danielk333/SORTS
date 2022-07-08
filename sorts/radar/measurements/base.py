import numpy as np
import ctypes

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

    def compute_measurement_jacobian(self, radar, txrx_pass, space_object, variables, deltas, transforms={}, **kwargs):
        if self.logger is not None:
            self.logger.debug(f'Measurement:compute_measurement_jacobian: variables={variables}, deltas={deltas}')

        if self.profiler is not None:
            self.profiler.start('Measurement:compute_measurement_jacobian')

        t = txrx_pass.start(), txrx_pass.end()

        if self.profiler is not None:
            self.profiler.start('Measurement:compute_measurement_jacobian:reference')

        data0 = self.calculate_observation(txrx_pass, t, generator, space_object, **kwargs)
        
        if data0 is None:
            return None, None

        keep = np.full(t.shape, False, dtype=np.bool)
        keep[data0['kept']] = True

        r = np.empty((len(t),2), dtype=np.float64)
        r_dot = np.empty((len(t),2), dtype=np.float64)

        r[keep,0] = data0['range']
        r_dot[keep,0] = data0['range_rate']


        if self.profiler is not None:
            self.profiler.stop('Measurement:compute_measurement_jacobian:reference')

        J = np.zeros([len(t)*2,len(variables)], dtype=np.float64)

        kwargs['snr_limit'] = False
        kwargs['calculate_snr'] = False

        for ind, var in enumerate(variables):
            if self.profiler is not None:
                self.profiler.start(f'Measurement:compute_measurement_jacobian:d_{var}')

            dso = space_object.copy()
            if var in transforms:
                Tx = transforms[var][0](getattr(dso, var)) + deltas[ind]
                dx = transforms[var][1](Tx)
            else:
                dx = getattr(dso, var) + deltas[ind]

            dso.update(**{var: dx})

            ddata = self.calculate_observation(txrx_pass, t, generator, dso, **kwargs)

            dkeep = np.full(t.shape, False, dtype=np.bool)
            dkeep[ddata['kept']] = True
            keep = np.logical_and(keep, dkeep)

            r[dkeep,1] = ddata['range']
            r_dot[dkeep,1] = ddata['range_rate']

            dr = (r[:,1] - r[:,0])/deltas[ind]
            dv = (r_dot[:,1] - r_dot[:,0])/deltas[ind]

            J[:len(t),ind]=dr
            J[len(t):,ind]=dv

            if self.profiler is not None:
                self.profiler.stop(f'Measurement:compute_measurement_jacobian:d_{var}')

        for key in data0:
            if key in ['kept']:
                continue
            elif isinstance(data0[key], np.ndarray):
                data0[key] = data0[key][...,keep[data0['kept']]]
            else:
                data0[key] = [x for ind_, x in enumerate(data0[key]) if keep[data0['kept']][ind_]]
        data0['kept'] = np.argwhere(keep).flatten()

        Jkeep = np.full((len(t)*2,), False, dtype=np.bool)
        Jkeep[:len(t)]=keep
        Jkeep[len(t):]=keep

        J = J[Jkeep,:]

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
        max_dpos=50e3,
        snr_limit=True, 
        save_states=False, 
        logger=None,
        profiler=None,
    ):
        if 'pointing_direction' not in radar_states.keys():
            raise ValueError("input radar_states are not valid. please provide radar_states containing beam directions")
        
        if profiler is not None: profiler.start('Measurements:Measure')

        # Checking input station indices
        tx_indices, rx_indices = self.__get_station_indices(tx_indices, rx_indices, radar)
        if logger is not None: logger.debug(f'Measurements:Measure: stations tx={tx_indices}, rx={rx_indices}')

        if not issubclass(interpolator, interpolation.Interpolator):
            raise TypeError(f"interpolator must be an instance of {interpolation.Interpolator}.")

        # checking radar_states
        if profiler is not None: profiler.start('Measurements:Measure:Initialization')
        n_periods = len(radar_states["t"])

        # get object states
        t_measurement               = np.ndarray((n_periods, len(tx_indices), len(rx_indices)), dtype=object)
        space_object_states         = np.ndarray((n_periods,), dtype=object)
        pass_mask                   = np.ndarray((n_periods,), dtype=object)

        for period_id in range(n_periods):
            pass_mask[period_id] = np.zeros((len(tx_indices), len(rx_indices) ,len(radar_states["pointing_direction"][period_id]["t"]),), dtype=np.int32)
            
            # t is always in scheduler relative time
            # t_samp is in space object relative time if there is a scheduler epoch, otherwise it is assumed that the epoch are the same
            # if there is an interpolator it is assumed that interpolation is done in space object relative time
            if epoch is None:
                t_samp = np.asfarray(radar_states["pointing_direction"][period_id]["t"])
            else:
                t_samp = np.asfarray(radar_states["pointing_direction"][period_id]["t"]) + (epoch - space_object.epoch).sec

            if profiler is not None: profiler.start('Measurements:Measure:Initialization:get_object_states')

            t_states = equidistant_sampling(
                orbit = space_object.state, 
                start_t = t_samp[0], 
                end_t = t_samp[-1], 
                max_dpos=max_dpos,
            )

            # propagate space object states
            states = space_object.get_state(t_states)
            if profiler is not None: profiler.stop('Measurements:Measure:Initialization:get_object_states')
            
            # 
            for txi in range(len(tx_indices)):
                for rxi in range(len(rx_indices)):
                    t_measurement[period_id][txi, rxi] = t_samp

                    if profiler is not None: profiler.start('Measurements:Measure:Initialization:find_passes')
                    passes = find_simultaneous_passes(t_states, states, [radar.tx[tx_indices[txi]], radar.rx[rx_indices[rxi]]])
                    if profiler is not None: profiler.stop('Measurements:Measure:Initialization:find_passes')

                    # print("passes : ", passes)
                    # print("t_samp : ", t_samp)
                    # print("t_states : ", t_states)
                    if len(passes) > 0:
                        for _pass_id, _pass in enumerate(passes):
                            if _pass.inds[0] > 0:                   t_pass_start_id_lowres =  _pass.inds[0]-1
                            else:                                   t_pass_start_id_lowres =  _pass.inds[0]
                            if _pass.inds[-1] < len(t_states)-1:     t_pass_end_id_lowres =   _pass.inds[-1]+1
                            else:                                   t_pass_end_id_lowres =    _pass.inds[-1]

                            pass_mask_tmp = np.where(np.logical_and(t_samp >= t_states[t_pass_start_id_lowres], t_samp <= t_states[t_pass_end_id_lowres]))[0]
                            del t_pass_end_id_lowres, t_pass_start_id_lowres

                            if len(pass_mask_tmp) > 0:
                                t_pass_start_id = pass_mask_tmp[0]
                                t_pass_end_id = pass_mask_tmp[-1]

                                pass_mask[period_id][txi, rxi][t_pass_start_id:t_pass_end_id] = 1
                
            space_object_states[period_id] = interpolator(states, t_states).get_state(t_samp)
            # print("pass_mask, ", pass_mask)
            
        # Initializing computation results
        if calculate_snr:
            diameter        = space_object.d
            spin_period     = space_object.parameters.get('spin_period', 0.0)
            radar_albedo    = space_object.parameters.get('radar_albedo', 1.0)

            snr             = np.ndarray((n_periods,), dtype=object)
            snr_inch        = np.ndarray((n_periods,), dtype=object)
            rcs             = np.ndarray((n_periods,), dtype=object)
        else:
            snr = None
            snr_inch = None
            rcs = None

        if profiler is not None: profiler.stop('Measurements:Measure:Initialization')

        # compute measurements for each scheduler period
        for period_id in range(n_periods):
            n_dirs                      = len(radar_states["pointing_direction"][period_id]["t"])
            n_time_slices               = len(radar_states["t"][period_id])

            snr[period_id]              = np.ndarray((len(tx_indices), len(rx_indices),), dtype=object)
            snr_inch[period_id]         = np.ndarray((len(tx_indices), len(rx_indices),), dtype=object)
            rcs[period_id]              = np.ndarray((len(tx_indices), n_time_slices,), dtype=np.float64)
            


            # compute range and range rates
            if profiler is not None: profiler.start('Measurements:Measure:enus,range,range_rate')
            enu_tx_to_so, enu_rx_to_so, ranges_tx, ranges_rx, range_rates_tx, range_rates_rx = self.__compute_ranges(space_object_states[period_id], radar, tx_indices, rx_indices)
            if profiler is not None: profiler.stop('Measurements:Measure:enus,range,range_rate')

            # compute beam gains
            if profiler is not None: profiler.start('Measurements:Measure:beam_gain')
            tx_gain, rx_gain = self.compute_gain(radar, radar_states, enu_tx_to_so, enu_rx_to_so, pass_mask[period_id], tx_indices, rx_indices, period_id)
            if profiler is not None: profiler.stop('Measurements:Measure:beam_gain')

            if calculate_snr:
                for txi in tx_indices:
                    if profiler is not None: profiler.start('Measurements:Measure:rcs')
                    rcs[period_id][txi] = signals.hard_target_rcs(wavelength=radar_states["wavelength_tx"][period_id][txi], diameter=diameter)
                    if profiler is not None: profiler.start('Measurements:Measure:rcs')

                    for rxi in rx_indices:
                        keep                                = pass_mask[period_id][txi, rxi]

                        snr[period_id][txi, rxi]            = np.ndarray((n_dirs,), dtype=np.float64)                    
                        snr_inch[period_id][txi, rxi]       = np.ndarray((n_dirs,), dtype=np.float64)    

                        if profiler is not None: profiler.start('Measurements:Measure:snr')
                        clibsorts.compute_measurement_snr.argtypes = [
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["t"][period_id].ndim, shape=radar_states["t"][period_id].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["pointing_direction"][period_id]["t"].ndim, shape=radar_states["pointing_direction"][period_id]["t"].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["t_slice"][period_id].ndim, shape=radar_states["t_slice"][period_id].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["pulse_length"][period_id][txi].ndim, shape=radar_states["pulse_length"][period_id][txi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["ipp"][period_id][txi].ndim, shape=radar_states["ipp"][period_id][txi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=tx_gain[txi].ndim, shape=tx_gain[txi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=rx_gain[rxi].ndim, shape=rx_gain[rxi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["wavelength_tx"][period_id][txi].ndim, shape=radar_states["wavelength_tx"][period_id][txi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["power"][period_id][txi].ndim, shape=radar_states["power"][period_id][txi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=ranges_tx[txi].ndim, shape=ranges_tx[txi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=ranges_rx[rxi].ndim, shape=ranges_rx[rxi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["duty_cycle"][period_id][txi].ndim, shape=radar_states["duty_cycle"][period_id][txi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["coh_int_bandwidth"][period_id][txi].ndim, shape=radar_states["coh_int_bandwidth"][period_id][txi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["noise_temperature"][period_id][rxi].ndim, shape=radar_states["noise_temperature"][period_id][rxi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=snr[period_id][txi, rxi].ndim, shape=snr[period_id][txi, rxi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=snr_inch[period_id][txi, rxi].ndim, shape=snr_inch[period_id][txi, rxi].shape),
                            np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=keep.ndim, shape=keep.shape),
                            ctypes.c_double,
                            ctypes.c_double,
                            ctypes.c_double,
                            ctypes.c_double,
                            ctypes.c_int,
                            ctypes.c_int,
                            ctypes.c_int,
                            ]

                        clibsorts.compute_measurement_snr(
                            radar_states["t"][period_id],
                            radar_states["pointing_direction"][period_id]["t"],
                            radar_states["t_slice"][period_id],
                            radar_states["pulse_length"][period_id][txi],
                            radar_states["ipp"][period_id][txi],
                            tx_gain[txi],
                            rx_gain[rxi],
                            radar_states["wavelength_tx"][period_id][txi],
                            radar_states["power"][period_id][txi],
                            ranges_tx[txi],
                            ranges_rx[rxi],
                            radar_states["duty_cycle"][period_id][txi],
                            radar_states["coh_int_bandwidth"][period_id][txi],
                            radar_states["noise_temperature"][period_id][rxi],
                            snr[period_id][txi, rxi],
                            snr_inch[period_id][txi, rxi],
                            keep,
                            ctypes.c_double(diameter),
                            ctypes.c_double(radar_albedo),
                            ctypes.c_double(spin_period),
                            ctypes.c_double(radar.min_SNRdb),
                            ctypes.c_int(doppler_spread_integrated_snr),
                            ctypes.c_int(snr_limit),
                            ctypes.c_int(n_dirs),
                            )
                        if profiler is not None: profiler.start('Measurements:Measure:snr')

                        keep = keep.astype(bool)

                        snr[period_id][txi, rxi] = snr[period_id][txi, rxi][keep]
                        snr_inch[period_id][txi, rxi] = snr_inch[period_id][txi, rxi][keep]
                        t_measurement[period_id][txi, rxi] = t_measurement[period_id][txi, rxi][keep]

        data = dict(
            t = t_measurement,
            snr = snr,
            range = ranges_tx + ranges_rx,
            range_rate = range_rates_tx + range_rates_rx,
            pointing_direction = radar_states["pointing_direction"],
            rcs = rcs,
        )
        if doppler_spread_integrated_snr:
            data['snr_inch'] = snr_inch
        if save_states:
            data['states'] = space_object_states

        if profiler is not None: profiler.stop('Measurements:Measure')

        if logger is not None:
            logger.debug(f'Measure:completed')

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


    def compute_gain(self, radar, radar_states, tx_enus, rx_enus, pass_mask, tx_indices, rx_indices, period_id):
        '''
        Given the input beam configured by the controller and the local (for that beam) coordinates of the observed object, get the correct gain and wavelength used for SNR calculation. 

        The default is a maximum calculation based on pointing, this is used for e.g. RX digital beam-forming.
        '''
        N = len(radar_states["pointing_direction"][period_id]["t"])
        
        gain_tx = np.ndarray((len(tx_indices), N,), dtype=np.float64)
        gain_rx = np.ndarray((len(rx_indices), len(tx_indices), N,), dtype=np.float64)

        def compute_antenna_gain_tx(txi, tdirs_id, t_id):
            nonlocal radar, radar_states, tx_enus, tx_indices, period_id
            tx_id = tx_indices[txi]

            radar.tx[tx_id].wavelength = radar_states["wavelength_tx"][period_id][txi, t_id]
            radar.tx[tx_id].point(radar_states["pointing_direction"][period_id]["tx"][txi, 0, :, t_id])

            gain = 1.0
            if len(radar.tx[tx_id].beam.pointing.shape) > 1:
                gain = np.max([
                    radar.tx[tx_id].beam.gain(tx_enus[txi, 0:3, tdirs_id], ind={'pointing': pi})
                    for pi in range(radar.tx[tx_id].beam.pointing.shape[1])
                ])
            else:
                gain = radar.tx[tx_id].beam.gain(tx_enus[txi, 0:3, tdirs_id])

            return gain

        def compute_antenna_gain_rx(rxi, txi, tdirs_id, t_id):
            nonlocal radar, radar_states, rx_enus, rx_indices, period_id
            rx_id = rx_indices[rxi]

            radar.rx[rx_id].wavelength = radar_states["wavelength_rx"][period_id][rxi, t_id]
            radar.rx[rx_id].point(radar_states["pointing_direction"][period_id]["rx"][rxi, txi, :, t_id])

            gain = 1.0
            if len(radar.rx[rx_id].beam.pointing.shape) > 1:
                gain = np.max([
                    radar.rx[rx_id].beam.gain(rx_enus[rxi, 0:3, tdirs_id], ind={'pointing': pi})
                    for pi in range(radar.rx[rx_id].beam.pointing.shape[1])
                ])
            else:
                gain = radar.rx[rx_id].beam.gain(rx_enus[rxi, 0:3, tdirs_id])

            return gain

        # define callback functions 
        COMPUTE_GAIN_TX = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int)
        COMPUTE_GAIN_RX = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int)

        compute_antenna_gain_tx_c = COMPUTE_GAIN_TX(compute_antenna_gain_tx)
        compute_antenna_gain_rx_c = COMPUTE_GAIN_RX(compute_antenna_gain_rx)

        clibsorts.compute_gain.argtypes = [
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["t"][period_id].ndim, shape=radar_states["t"][period_id].shape),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=radar_states["pointing_direction"][period_id]["t"].ndim, shape=radar_states["pointing_direction"][period_id]["t"].shape),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=gain_tx.ndim, shape=gain_tx.shape),
            np.ctypeslib.ndpointer(dtype=ctypes.c_double, ndim=gain_rx.ndim, shape=gain_rx.shape),
            np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=pass_mask.ndim, shape=pass_mask.shape),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            COMPUTE_GAIN_TX,
            COMPUTE_GAIN_RX,
            ]

        print("t, ", radar_states["t"][period_id].dtype)
        print("pointing_direction, ", radar_states["pointing_direction"][period_id]["t"].dtype)
        clibsorts.compute_gain(
            radar_states["t"][period_id],
            radar_states["pointing_direction"][period_id]["t"],
            gain_tx, 
            gain_rx,
            pass_mask.astype(np.int32),
            ctypes.c_int(len(radar_states["pointing_direction"][period_id]["t"])),
            ctypes.c_int(len(tx_indices)),
            ctypes.c_int(len(rx_indices)),
            compute_antenna_gain_tx_c,
            compute_antenna_gain_rx_c,
            )
        print("gain_tx, ", gain_tx)

        return gain_tx, gain_rx