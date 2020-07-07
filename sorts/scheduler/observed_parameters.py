#!/usr/bin/env python

'''

'''

import numpy as np

from .scheduler import Scheduler

from ..signals import hard_target_snr, hard_target_rcs
from ..passes import Pass


class ObservedParameters(Scheduler):
    '''Bi-static radar observation parameters of hard targets.
    ''' 

    def __init__(self, radar, logger=None, profiler=None, **kwargs):
        super().__init__(
            radar=radar, 
            logger=logger, 
            profiler=profiler,
        )


    def calculate_observation(self, txrx_pass, t, generator, space_object, remove_failed=True):
        txi, rxi = txrx_pass.station_id

        if self.logger is not None:
            self.logger.info(f'Obs.Param.:calculate_observation:(tx={rxi}, rx={rxi}), len(t) = {len(t)}')

        if self.profiler is not None:
            self.profiler.start('Obs.Param.:calculate_observation')

        if self.profiler is not None:
            self.profiler.start('Obs.Param.:calculate_observation:get_state')
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

        for ti, radar in enumerate(generator):
            if self.profiler is not None:
                self.profiler.start('Obs.Param.:calculate_observation:snr-step')

            if radar.tx[txi].enabled and radar.rx[rxi].enabled:

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
                    diameter=space_object.d,
                    bandwidth=radar.tx[txi].coh_int_bandwidth,
                    rx_noise_temp=radar.rx[rxi].noise,
                )
                if self.profiler is not None:
                    self.profiler.stop('Obs.Param.:calculate_observation:snr-step:snr')
                
                rcs[ti] = hard_target_rcs(
                    wavelength=radar.rx[rxi].wavelength,
                    diameter=space_object.d,
                )

                keep[ti] = np.log10(snr[ti])*10.0 > radar.min_SNRdb

            else:
                snr[ti] = np.nan
                rcs[ti] = np.nan
                keep[ti] = False
            
            if self.profiler is not None:
                self.profiler.stop('Obs.Param.:calculate_observation:snr-step')

        data = dict(
            t = t,
            snr = snr,
            range = ranges[0] + ranges[1],
            range_rate = range_rates[0] + range_rates[1],
            tx_k = enus[0][:3,:]/ranges[0],
            rx_k = enus[1][:3,:]/ranges[1],
            rcs = rcs,
        )

        if remove_failed:
            if np.any(keep):
                for key in data:
                    data[key] = data[key][...,keep]
            else:
                data = None
        if self.profiler is not None:
            self.profiler.stop('Obs.Param.:calculate_observation')

        if self.logger is not None:
            self.logger.info(f'Obs.Param.:calculate_observation:complete')
        return data