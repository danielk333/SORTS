#!/usr/bin/env python

'''

'''

import numpy as np

from .radar_controller import RadarController


class Tracker(RadarController):
    '''Takes in ECEF points and a time vector and creates a tracking control.
    '''

    def __init__(self, radar, t, ecefs):
        super().__init__(radar)
        self.ecefs = ecefs
        self.t = t

    def point_radar(self, ind):
        self.point_tx_ecef(self.ecefs[:3,ind])
        self.point_rx_ecef(self.ecefs[:3,ind])
        return self.radar


    def generator(self, t):
        for ti in range(len(t)):
            ind = np.argmax(t[ti] <= self.t)
            yield self.point_radar(ind)

