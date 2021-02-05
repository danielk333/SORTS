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


    def calculate_observation(self, txrx_pass, t, generator, space_object, calculate_snr=True, interpolator=None, snr_limit=True):
        '''Calculate the observation of a pass of a specific space object given the current state of the Scheduler.

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
        
        if interpolator is not None:
            states = interpolator.get_state(t)
        else:
            states = space_object.get_state(t)

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
        for ti, mrad in enumerate(generator):
            radar, meta = mrad
            metas.append(meta)
            if self.profiler is not None:
                self.profiler.start('Obs.Param.:calculate_observation:snr-step')

            if radar.tx[txi].enabled and radar.rx[rxi].enabled and calculate_snr:

                if self.profiler is not None:
                    self.profiler.start('Obs.Param.:calculate_observation:snr-step:gain')
                if len(radar.tx[txi].beam.pointing.shape) > 1:
                    tx_g = np.max([
                        radar.tx[txi].beam.gain(enus[0][:3,ti], ind={'pointing': pi})
                        for pi in range(radar.tx[txi].beam.pointing.shape[1])
                    ])
                else:
                    tx_g = radar.tx[txi].beam.gain(enus[0][:3,ti])

                if len(radar.rx[rxi].beam.pointing.shape) > 1:
                    rx_g = np.max([
                        radar.rx[rxi].beam.gain(enus[1][:3,ti], ind={'pointing': pi})
                        for pi in range(radar.rx[rxi].beam.pointing.shape[1])
                    ])
                else:
                    rx_g = radar.rx[rxi].beam.gain(enus[1][:3,ti])

                if self.profiler is not None:
                    self.profiler.stop('Obs.Param.:calculate_observation:snr-step:gain')
                    self.profiler.start('Obs.Param.:calculate_observation:snr-step:snr')

                snr[ti] = hard_target_snr(
                    tx_g,
                    rx_g,
                    radar.rx[rxi].wavelength,
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
                    wavelength=radar.rx[rxi].wavelength,
                    diameter=diam,
                )
                if snr_limit:
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