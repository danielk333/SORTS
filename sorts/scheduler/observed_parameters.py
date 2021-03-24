#!/usr/bin/env python

'''

'''

import numpy as np
import scipy.constants

from .scheduler import Scheduler

from ..signals import hard_target_snr, hard_target_rcs
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
        
        snr_limit = kwargs.get('snr_limit', True)

        kwargs['snr_limit'] = False
        data0 = self.calculate_observation(txrx_pass, t, generator, space_object, **kwargs)

        snr_inds = np.full(data0['snr'].shape, True, dtype=np.bool)
        if snr_limit:
            snr_db = np.log10(data0['snr'])*10.0
            snr_inds[np.logical_or(np.isnan(snr_db),np.isinf(snr_db))] = False
            snr_inds[snr_inds] = snr_db[snr_inds] > self.radar.min_SNRdb

        for key in data0:
            if key == 'metas':
                data0[key] = [x for i, x in enumerate(data0[key]) if snr_inds[i]]
            else:
                data0[key] = data0[key][...,snr_inds]

        if self.profiler is not None:
            self.profiler.stop('Obs.Param.:calculate_observation_jacobian:reference')

        if data0 is None:
            return None, None

        t_filt = t[snr_inds]
        J = np.zeros([len(t_filt)*2,len(variables)], dtype=np.float64)

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
            for key in data0:
                if key == 'metas':
                    continue

                ddata[key] = ddata[key][...,snr_inds]

            dr = (ddata['range'] - data0['range'])/deltas[ind]
            dv = (ddata['range_rate'] - data0['range_rate'])/deltas[ind]

            J[:len(t_filt),ind]=dr
            J[len(t_filt):,ind]=dv

            if self.profiler is not None:
                self.profiler.stop(f'Obs.Param.:calculate_observation_jacobian:d_{var}')

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
            bandwidths = np.empty((len(t),), dtype=np.float64)
            rx_noise_temps = np.empty((len(t),), dtype=np.float64)
            txrx_on = np.full((len(t),), False, dtype=np.bool)

            vectorized_data = None
            for ri, (radar, meta) in enumerate(generator):
                if extended_meta:
                    self.extend_meta(t[ri], txi, rxi, radar, meta)

                metas.append(meta)
                vec_row = self.get_vectorized_row(radar, meta, txi, rxi)
                if vectorized_data is None:
                    vectorized_data = np.empty((len(t), len(vec_row)), dtype=vec_row.dtype)
                
                vectorized_data[ri,:] = vec_row

                powers[ri] = radar.tx[txi].power
                bandwidths[ri] = radar.tx[txi].coh_int_bandwidth
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

                snr[keep] = hard_target_snr(
                    tx_g,
                    rx_g,
                    tx_wavelength,
                    powers[keep],
                    ranges[0][keep],
                    ranges[1][keep],
                    diameter=diam,
                    bandwidth=bandwidths[keep],
                    rx_noise_temp=rx_noise_temps[keep],
                )
                if self.profiler is not None:
                    self.profiler.stop('Obs.Param.:calculate_observation:snr-step:snr')
                    self.profiler.start('Obs.Param.:calculate_observation:snr-step:rcs,filter')

                rcs[keep] = hard_target_rcs(
                    wavelength=tx_wavelength,
                    diameter=diam,
                )
                if snr_limit:
                    snr_db = np.empty(snr.shape, dtype=np.float64)
                    snr_db[snr < 1e-9] = np.nan
                    snr_db[snr >= 1e-9] = np.log10(snr[snr >= 1e-9])*10.0

                    keep[np.logical_or(np.isnan(snr_db), np.isinf(snr_db))] = False
                    #todo: fix this so it does not compare the already discarded values
                    keep[np.logical_and(snr_db <= radar.min_SNRdb, keep)] = False

                if self.profiler is not None:
                    self.profiler.stop('Obs.Param.:calculate_observation:snr-step:rcs,filter')

        else:
            for ti, (radar, meta) in enumerate(generator):
                if extended_meta:
                    self.extend_meta(t[ti], txi, rxi, radar, meta)

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

                if self.profiler is not None:
                    self.profiler.stop('Obs.Param.:calculate_observation:observable_filter')
                    self.profiler.start('Obs.Param.:calculate_observation:snr-step')

                if radar.tx[txi].enabled and radar.rx[rxi].enabled and calculate_snr and observable:

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

                    snr[ti] = hard_target_snr(
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

                    rcs[ti] = hard_target_rcs(
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
                    if calculate_snr:
                        keep[ti] = False
                
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
        else:
            data = None

        if self.profiler is not None:
            self.profiler.stop('Obs.Param.:calculate_observation')

        if self.logger is not None:
            self.logger.debug(f'Obs.Param.:calculate_observation:complete')
        return data