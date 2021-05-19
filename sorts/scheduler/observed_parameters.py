#!/usr/bin/env python

'''

'''

import numpy as np
import scipy.constants

from .scheduler import Scheduler

from .. import signals
from ..passes import Pass


class ObservedParameters(Scheduler):
    '''Bi-static radar observation parameters of hard targets.

    **Parameters calculated**

        * time
        * signal to noise ratio
        * range
        * range rate
        * transmitter local pointing to target k
        * receiver pointing to target k
        * radar cross section

        #TODO: Docstring

    ''' 

    def __init__(self, radar, logger=None, profiler=None, **kwargs):
        super().__init__(
            radar=radar, 
            logger=logger, 
            profiler=profiler,
        )


    def calculate_observation_jacobian(self, txrx_pass, space_object, variables, deltas, transforms={}, **kwargs):
        '''Calculate the observation and its Jacobean of a pass of a specific space object given the current state of the Scheduler. 

        The Jacobean assumes that the SpaceObject has a Orbit state. To perturb non Orbit states a custom implementation is needed.

        NOTE: During the numerical calculation of the Jacobean only the range and range rates are calculated and `calculate_snr=False`.

        #TODO: Docstring
        '''

        if self.logger is not None:
            self.logger.debug(f'Obs.Param.:calculate_observation_jacobian:{variables}, deltas={deltas}')

        if self.profiler is not None:
            self.profiler.start('Obs.Param.:calculate_observation_jacobian')

        t, generator = self(txrx_pass.start(), txrx_pass.end())
        if generator is None:
            return None, None

        if self.profiler is not None:
            self.profiler.start('Obs.Param.:calculate_observation_jacobian:reference')
        
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
            self.profiler.stop('Obs.Param.:calculate_observation_jacobian:reference')

        J = np.zeros([len(t)*2,len(variables)], dtype=np.float64)

        kwargs['snr_limit'] = False
        kwargs['calculate_snr'] = False
        for ind, var in enumerate(variables):
            if self.profiler is not None:
                self.profiler.start(f'Obs.Param.:calculate_observation_jacobian:d_{var}')

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
                self.profiler.stop(f'Obs.Param.:calculate_observation_jacobian:d_{var}')

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
            self.profiler.stop('Obs.Param.:calculate_observation_jacobian')

        return data0, J


    def get_beam_gain_and_wavelength(self, beam, enu, meta):
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

        return g, beam.wavelength


    def stop_condition(self, t, radar, meta, tx_enu, rx_enu, tx_index, rx_index):
        '''Implement this function if there should be abort condition in the sequential generator. Work with both vectorization and linear iteration (as vectorization first caches the generator).
        '''
        return False


    def observable_filter(self, t, radar, meta, tx_enu, rx_enu, tx_index, rx_index):
        '''Determines if the object was observable or not by the radar system
        '''
        return True


    def vectorized_get_beam_gain_and_wavelength(self, t, vectorized_data, tx_enus, rx_enus, tx_index, rx_index):
        tx_g = self.radar.tx[tx_index].beam.gain(
            tx_enus, 
            pointing=vectorized_data[:,0:3].T, 
            vectorized_parameters=True,
        )
        rx_g = self.radar.rx[rx_index].beam.gain(
            rx_enus, 
            pointing=vectorized_data[:,4:7].T, 
            vectorized_parameters=True,
        )

        return tx_g, vectorized_data[:,3], rx_g, vectorized_data[:,7]


    def vectorized_observable_filter(self, t, vectorized_data, tx_enus, rx_enus, tx_index, rx_index):
        return np.full(t.shape, True, dtype=np.bool)


    def get_vectorized_row(self, radar, meta, tx_index, rx_index):
        '''Used to extract the used data from the `radar` instance when vectorizing `calculate_observation`. 
        Input to `vectorized_observable_filter` and `vectorized_get_beam_gain_and_wavelength`.

        Should return a numpy vector, will be stored as a row-vector in the matrix passed to the vectorized functions.
        '''
        row = np.empty((8,), dtype=np.float64)
        row[0:3] = radar.tx[tx_index].beam.pointing[:]
        row[3] = radar.tx[tx_index].beam.wavelength
        row[4:7] = radar.rx[rx_index].beam.pointing[:]
        row[7] = radar.rx[rx_index].beam.wavelength
        return row

    def extend_meta(self, t, txi, rxi, radar, meta):
        meta['pulse_length'] = radar.tx[txi].pulse_length
        meta['ipp'] = radar.tx[txi].ipp
        meta['n_ipp'] = radar.tx[txi].n_ipp


    def calculate_observation(
            self, 
            txrx_pass, 
            t, 
            generator, 
            space_object, 
            epoch=None, 
            calculate_snr=True, 
            doppler_spread_integrated_snr=False,
            interpolator=None, 
            snr_limit=True, 
            save_states=False, 
            vectorize=False,
            extended_meta=True,
        ):
        '''Calculate the observation of a pass of a specific space object given the current state of the Scheduler.

        #ASSUMES INPUT t IS RELATIVE SPACE OBJECT EPOCH unless epoch is given

        #TODO: Docstring
        '''

        txi, rxi = txrx_pass.station_id

        if self.logger is not None:
            self.logger.debug(f'Obs.Param.:calculate_observation:(tx={txi}, rx={rxi}), len(t) = {len(t)}')

        if self.profiler is not None:
            self.profiler.start('Obs.Param.:calculate_observation')

        if calculate_snr:
            diam = space_object.d
        else:
            diam = None

        spin_period = space_object.parameters.get('spin_period', None)
        radar_albedo = space_object.parameters.get('radar_albedo', 1.0)

        if self.profiler is not None:
            self.profiler.start('Obs.Param.:calculate_observation:get_state')
        
        #t is always in scheduler relative time
        #t_samp is in space object relative time if there is a scheduler epoch, otherwise it is assumed that the epoch are the same
        #if there is an interpolator it is assumed that interpolation is done in space object relative time
        if epoch is None:
            t_samp = t
        else:
            t_samp = t + (epoch - space_object.epoch).sec

        if interpolator is not None:
            states = interpolator.get_state(t_samp)
        else:
            states = space_object.get_state(t_samp)

        if self.profiler is not None:
            self.profiler.stop('Obs.Param.:calculate_observation:get_state')

        snr = np.empty((len(t),), dtype=np.float64)
        snr_inch = np.empty((len(t),), dtype=np.float64)
        rcs = np.empty((len(t),), dtype=np.float64)
        keep = np.full((len(t),), True, dtype=np.bool)
        

        if self.profiler is not None:
            self.profiler.start('Obs.Param.:calculate_observation:enus,range,range_rate')

        enus = [
            self.radar.tx[txi].enu(states),
            self.radar.rx[rxi].enu(states),
        ]
        ranges = [Pass.calculate_range(enu) for enu in enus]
        range_rates = [Pass.calculate_range_rate(enu) for enu in enus]

        if self.profiler is not None:
            self.profiler.stop('Obs.Param.:calculate_observation:enus,range,range_rate')
            self.profiler.start('Obs.Param.:calculate_observation:generator')

        metas = []

        if vectorize:

            powers = np.empty((len(t),), dtype=np.float64)
            t_slices = np.empty((len(t),), dtype=np.float64)
            pulse_lengths = np.empty((len(t),), dtype=np.float64)
            ipps = np.empty((len(t),), dtype=np.float64)
            bandwidths = np.empty((len(t),), dtype=np.float64)
            duty_cycles = np.empty((len(t),), dtype=np.float64)
            rx_noise_temps = np.empty((len(t),), dtype=np.float64)
            txrx_on = np.full((len(t),), False, dtype=np.bool)

            vectorized_data = None
            for ri, (radar, meta) in enumerate(generator):
                if extended_meta:
                    self.extend_meta(t[ri], txi, rxi, radar, meta)

                stop = self.stop_condition(
                    t[ri], 
                    radar, 
                    meta, 
                    enus[0][:3,ri], 
                    enus[1][:3,ri], 
                    txi, 
                    rxi,
                )
                if stop:
                    break

                metas.append(meta)
                vec_row = self.get_vectorized_row(radar, meta, txi, rxi)
                if vectorized_data is None:
                    vectorized_data = np.empty((len(t), len(vec_row)), dtype=vec_row.dtype)
                
                vectorized_data[ri,:] = vec_row

                t_slice_ = meta.get('t_slice', None)
                if t_slice_ is not None:
                    t_slices[ri] = t_slice_
                else:
                    t_slices[ri] = np.nan
                pulse_lengths[ri] = radar.tx[txi].pulse_length
                ipps[ri] = radar.tx[txi].ipp
                powers[ri] = radar.tx[txi].power
                bandwidths[ri] = radar.tx[txi].coh_int_bandwidth
                duty_cycles[ri] = radar.tx[txi].duty_cycle
                rx_noise_temps[ri] = radar.rx[rxi].noise

                if radar.tx[txi].enabled and radar.rx[rxi].enabled:
                    txrx_on[ri] = True

            if self.profiler is not None:
                self.profiler.stop('Obs.Param.:calculate_observation:generator')
                self.profiler.start('Obs.Param.:calculate_observation:vectorized_observable_filter,keep')

            observable = self.vectorized_observable_filter(
                t, 
                vectorized_data,
                enus[0][:3,:], 
                enus[1][:3,:], 
                txi, 
                rxi,
            )
            keep = np.logical_and(observable, txrx_on)

            if self.profiler is not None:
                self.profiler.stop('Obs.Param.:calculate_observation:vectorized_observable_filter,keep')
                self.profiler.start('Obs.Param.:calculate_observation:snr-step')

            if calculate_snr:
                if self.profiler is not None:
                    self.profiler.start('Obs.Param.:calculate_observation:snr-step:gain')

                tx_g, tx_wavelength, rx_g, rx_wavelength = self.vectorized_get_beam_gain_and_wavelength(
                    t[keep], 
                    vectorized_data[keep,:], 
                    enus[0][:3,keep], 
                    enus[1][:3,keep], 
                    txi, 
                    rxi,
                )
                
                if self.profiler is not None:
                    self.profiler.stop('Obs.Param.:calculate_observation:snr-step:gain')
                    self.profiler.start('Obs.Param.:calculate_observation:snr-step:snr')

                if doppler_spread_integrated_snr:
                    snr[keep], snr_inch[keep] = signals.doppler_spread_hard_target_snr(
                        t_slices[keep], 
                        spin_period, 
                        tx_g, 
                        rx_g,
                        tx_wavelength,
                        powers[keep],
                        ranges[0][keep], 
                        ranges[1][keep],
                        duty_cycle=duty_cycles[keep],
                        diameter=diam, 
                        bandwidth=bandwidths[keep],
                        rx_noise_temp=rx_noise_temps[keep],
                        radar_albedo=radar_albedo,
                    )
                else:
                    snr[keep] = signals.hard_target_snr(
                        tx_g,
                        rx_g,
                        tx_wavelength,
                        powers[keep],
                        ranges[0][keep],
                        ranges[1][keep],
                        diameter=diam,
                        bandwidth=bandwidths[keep],
                        rx_noise_temp=rx_noise_temps[keep],
                        radar_albedo=radar_albedo,
                    )

                snr_modulation = np.ones((len(t),), dtype=np.float64)
                for ch_txi, ch_rxi in self.radar.joint_stations:
                    if ch_rxi == rxi:
                        delay = (ranges[0][keep] + ranges[1][keep])/scipy.constants.c
                        ipp_f = np.mod(delay, ipps[keep])

                        inds = ipp_f <= pulse_lengths[keep]
                        snr_modulation[keep][inds] = pulse_lengths[keep][inds]

                        inds = ipp_f >= ipps[keep] - pulse_lengths[keep]
                        snr_modulation[keep][inds] = (ipps[keep][inds] - ipp_f[inds])/pulse_lengths[keep][inds]

                        break
                snr[keep] = snr[keep]*snr_modulation[keep]

                if self.profiler is not None:
                    self.profiler.stop('Obs.Param.:calculate_observation:snr-step:snr')
                    self.profiler.start('Obs.Param.:calculate_observation:snr-step:rcs,filter')

                rcs[keep] = signals.hard_target_rcs(
                    wavelength=tx_wavelength,
                    diameter=diam,
                )
                if snr_limit:
                    snr_db = np.empty(snr.shape, dtype=np.float64)
                    snr_db[snr < 1e-9] = np.nan
                    snr_db[snr >= 1e-9] = np.log10(snr[snr >= 1e-9])*10.0

                    keep[np.logical_or(np.isnan(snr_db), np.isinf(snr_db))] = False
                    keep[keep] = snr_db[keep] > radar.min_SNRdb

                if self.profiler is not None:
                    self.profiler.stop('Obs.Param.:calculate_observation:snr-step:rcs,filter')

        else:
            for ti, (radar, meta) in enumerate(generator):
                if extended_meta:
                    self.extend_meta(t[ti], txi, rxi, radar, meta)

                stop = self.stop_condition(
                    t[ti], 
                    radar, 
                    meta, 
                    enus[0][:3,ti], 
                    enus[1][:3,ti], 
                    txi, 
                    rxi,
                )
                if stop:
                    keep[ti:] = False
                    break

                metas.append(meta)

                if self.profiler is not None:
                    self.profiler.start('Obs.Param.:calculate_observation:observable_filter')

                observable = self.observable_filter(
                    t[ti], 
                    radar, 
                    meta, 
                    enus[0][:3,ti], 
                    enus[1][:3,ti], 
                    txi, 
                    rxi,
                )

                if not (radar.tx[txi].enabled and radar.rx[rxi].enabled and observable):
                    keep[ti] = False
                    continue

                if self.profiler is not None:
                    self.profiler.stop('Obs.Param.:calculate_observation:observable_filter')
                    self.profiler.start('Obs.Param.:calculate_observation:snr-step')

                if calculate_snr:

                    if self.profiler is not None:
                        self.profiler.start('Obs.Param.:calculate_observation:snr-step:gain')

                    #check if target is in radars blind range
                    #assume synchronized transmitting
                    #assume decoding of partial pulses is possible and linearly decreases signal strength
                    snr_modulation = 1.0
                    for ch_txi, ch_rxi in radar.joint_stations:
                        if ch_rxi == rxi:
                            delay = (ranges[0][ti] + ranges[1][ti])/scipy.constants.c
                            ipp_f = np.mod(delay, radar.tx[txi].ipp)

                            if ipp_f <= radar.tx[txi].pulse_length:
                                snr_modulation = ipp_f/radar.tx[txi].pulse_length
                            elif ipp_f >= radar.tx[txi].ipp - radar.tx[txi].pulse_length:
                                snr_modulation = (radar.tx[txi].ipp - ipp_f)/radar.tx[txi].pulse_length
                            break

                    tx_g, tx_wavelength = self.get_beam_gain_and_wavelength(
                        radar.tx[txi].beam, 
                        enus[0][:3,ti], 
                        meta,
                    )
                    rx_g, rx_wavelength = self.get_beam_gain_and_wavelength(
                        radar.rx[rxi].beam, 
                        enus[1][:3,ti], 
                        meta,
                    )
                    
                    if self.profiler is not None:
                        self.profiler.stop('Obs.Param.:calculate_observation:snr-step:gain')
                        self.profiler.start('Obs.Param.:calculate_observation:snr-step:snr')

                    if doppler_spread_integrated_snr:
                        snr[keep], snr_inch[keep] = signals.doppler_spread_hard_target_snr(
                            meta.get('t_slice', np.nan), 
                            spin_period, 
                            tx_g, 
                            rx_g,
                            tx_wavelength,
                            radar.tx[txi].power,
                            ranges[0][ti], 
                            ranges[1][ti],
                            duty_cycle=radar.tx[txi].duty_cycle,
                            diameter=diam, 
                            bandwidth=radar.tx[txi].coh_int_bandwidth,
                            rx_noise_temp=radar.rx[rxi].noise,
                            radar_albedo=radar_albedo,
                        )
                    else:
                        snr[ti] = signals.hard_target_snr(
                            tx_g,
                            rx_g,
                            tx_wavelength,
                            radar.tx[txi].power,
                            ranges[0][ti],
                            ranges[1][ti],
                            diameter=diam,
                            bandwidth=radar.tx[txi].coh_int_bandwidth,
                            rx_noise_temp=radar.rx[rxi].noise,
                        )
                    snr[ti] *= snr_modulation
                    if self.profiler is not None:
                        self.profiler.stop('Obs.Param.:calculate_observation:snr-step:snr')
                        self.profiler.start('Obs.Param.:calculate_observation:snr-step:rcs,filter')

                    rcs[ti] = signals.hard_target_rcs(
                        wavelength=tx_wavelength,
                        diameter=diam,
                    )
                    if snr_limit:
                        if snr[ti] < 1e-9:
                            snr_db = np.nan
                        else:
                            snr_db = np.log10(snr[ti])*10.0

                        if np.isnan(snr_db) or np.isinf(snr_db):
                            keep[ti] = False
                        else:
                            keep[ti] = snr_db > radar.min_SNRdb

                    if self.profiler is not None:
                        self.profiler.stop('Obs.Param.:calculate_observation:snr-step:rcs,filter')
                else:
                    snr[ti] = np.nan
                    rcs[ti] = np.nan
                
                if self.profiler is not None:
                    self.profiler.stop('Obs.Param.:calculate_observation:snr-step')
            if self.profiler is not None:
                self.profiler.stop('Obs.Param.:calculate_observation:generator')

        data = dict(
            t = t,
            snr = snr,
            range = ranges[0] + ranges[1],
            range_rate = range_rates[0] + range_rates[1],
            tx_k = enus[0][:3,:]/ranges[0],
            rx_k = enus[1][:3,:]/ranges[1],
            rcs = rcs,
            metas = metas,
        )
        if save_states:
            data['states'] = states

        if np.any(keep):
            for key in data:
                if isinstance(data[key], np.ndarray):
                    data[key] = data[key][...,keep]
                else:
                    data[key] = [x for ind_, x in enumerate(data[key]) if keep[ind_]]
            data['kept'] = np.argwhere(keep).flatten()
        else:
            data = None

        if self.profiler is not None:
            self.profiler.stop('Obs.Param.:calculate_observation')

        if self.logger is not None:
            self.logger.debug(f'Obs.Param.:calculate_observation:complete')
        return data