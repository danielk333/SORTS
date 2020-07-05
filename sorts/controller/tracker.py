#!/usr/bin/env python

'''

'''

import numpy as np

from .radar_controller import RadarController


class Tracker(RadarController):
    '''Takes in ECEF points and a time vector and creates a tracking control.
    '''

    def __init__(self, radar, t, ecefs, t0=0.0, dwell=0.1):
        super().__init__(radar, t=t, t0=t0)
        self.ecefs = ecefs
        self.dwell

    def point_radar(self, ind):
        self.point_tx_ecef(self.ecefs[:3,ind])
        self.point_rx_ecef(self.ecefs[:3,ind])


    def generator(self, t):
        for ti in range(len(t)):
            dt = t[ti] - self.t
            check = np.logical_and(dt > 0, dt < self.dwell)

            if np.any(check):
                ind = np.argmax(check)
                self.turn_on()
                self.point_radar(ind)
            else:
                self.turn_off()

            yield self.radar

