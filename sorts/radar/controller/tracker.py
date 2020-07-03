#!/usr/bin/env python

'''

'''

import numpy as np

from .radar_controller import RadarController


class Tracker(RadarController):
    '''Takes in ECEF points and a time vector and creates a tracking control.
    '''

    def __init__(self, radar, t, ecefs, t0=0.0):
        super().__init__(radar, t=t, t0=t0)
        self.ecefs = ecefs

    def point_radar(self, ind):
        self.point_tx_ecef(self.ecefs[:3,ind])
        self.point_rx_ecef(self.ecefs[:3,ind])
        return self.radar


    def generator(self, t):
        for ti in range(len(t)):
            ind = np.argmax(t[ti] <= self.t)
            yield self.point_radar(ind)

