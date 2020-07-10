#!/usr/bin/env python

'''

'''

import numpy as np

from .radar_controller import RadarController


class Tracker(RadarController):
    '''Takes in ECEF points and a time vector and creates a tracking control.
    '''
    
    META_FIELDS = RadarController.META_FIELDS + [
        'dwell',
        'target',
    ]

    def __init__(self, radar, t, ecefs, t0=0.0, dwell=0.1, return_copy=False, profiler=None, logger=None, meta=None):
        super().__init__(radar, t=t, t0=t0, profiler=profiler, logger=logger, meta=meta)
        self.ecefs = ecefs
        self.dwell = dwell
        self.return_copy = return_copy

        if self.logger is not None:
            self.logger.info(f'Tracker:init')

    def default_meta(self):
        dic = super().default_meta()
        dic['dwell'] = self.dwell
        return dic

    def point_radar(self, radar, ind):
        if self.profiler is not None:
            self.profiler.start('Tracker:generator:point_radar')

        RadarController.point_tx_ecef(radar, self.ecefs[:3,ind])
        RadarController.point_rx_ecef(radar, self.ecefs[:3,ind])

        if self.profiler is not None:
            self.profiler.stop('Tracker:generator:point_radar')

    def generator(self, t):
        if self.profiler is not None:
            self.profiler.start('Tracker:generator')
        if self.logger is not None:
            self.logger.debug(f'Tracker:generator: len(t) = {len(t)}')

        for ti in range(len(t)):
            if self.profiler is not None:
                self.profiler.start('Tracker:generator:step')

            if self.return_copy:
                radar = self.radar.copy()
            else:
                radar = self.radar

            dt = t[ti] - self.t
            check = np.logical_and(dt >= 0, dt <= self.dwell)

            if np.any(check):
                ind = np.argmax(check)
                RadarController.turn_on(radar)
                self.point_radar(radar, ind)
            else:
                RadarController.turn_off(radar)

            if self.profiler is not None:
                self.profiler.stop('Tracker:generator:step')

            yield radar, self.default_meta()

        if self.profiler is not None:
            self.profiler.stop('Tracker:generator')
        if self.logger is not None:
            self.logger.debug(f'Tracker:generator:completed')
