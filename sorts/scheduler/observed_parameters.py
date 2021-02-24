#!/usr/bin/env python

'''

'''

import numpy as np

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


    def get_beam_gain_wavelength(self, beam, enu, meta):
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


    def calculate_observation(self, txrx_pass, t, generator, space_object, epoch=None, calculate_snr=True, interpolator=None, snr_limit=True, save_states=False):
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

        if self.profiler is not None:
            self.profiler.start('Obs.Param.:calculate_observation:generator')
        metas = []
        for ti, (radar, meta) in enumerate(generator):

            metas.append(meta)
            if self.profiler is not None:
                self.profiler.start('Obs.Param.:calculate_observation:snr-step')

            if radar.tx[txi].enabled and radar.rx[rxi].enabled and calculate_snr:

                if self.profiler is not None:
                    self.profiler.start('Obs.Param.:calculate_observation:snr-step:gain')

                tx_g, tx_wavelength = self.get_beam_gain_wavelength(radar.tx[txi].beam, enus[0][:3,ti], meta)
                rx_g, rx_wavelength = self.get_beam_gain_wavelength(radar.rx[rxi].beam, enus[1][:3,ti], meta)
                
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
                if self.profiler is not None:
                    self.profiler.stop('Obs.Param.:calculate_observation:snr-step:snr')

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