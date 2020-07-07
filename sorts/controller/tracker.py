#!/usr/bin/env python

'''

'''

import numpy as np

from .radar_controller import RadarController


class Tracker(RadarController):
    '''Takes in ECEF points and a time vector and creates a tracking control.
    '''

    def __init__(self, radar, t, ecefs, t0=0.0, dwell=0.1, profiler=None, logger=None):
        super().__init__(radar, t=t, t0=t0, profiler=profiler, logger=logger)
        self.ecefs = ecefs
        self.dwell = dwell

        if self.logger is not None:
            self.logger.info(f'Tracker:init')

    def point_radar(self, ind):
        if self.profiler is not None:
            self.profiler.start('Tracker:generator:point_radar')

        self.point_tx_ecef(self.ecefs[:3,ind])
        self.point_rx_ecef(self.ecefs[:3,ind])

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

            dt = t[ti] - self.t
            check = np.logical_and(dt >= 0, dt <= self.dwell)

            if np.any(check):
                ind = np.argmax(check)
                self.turn_on()
                self.point_radar(ind)
            else:
                self.turn_off()

            if self.profiler is not None:
                self.profiler.stop('Tracker:generator:step')

            yield self.radar

        if self.profiler is not None:
            self.profiler.stop('Tracker:generator')
        if self.logger is not None:
            self.logger.debug(f'Tracker:generator:completed')
