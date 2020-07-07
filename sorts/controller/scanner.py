#!/usr/bin/env python

'''

'''

import numpy as np

from .radar_controller import RadarController


class Scanner(RadarController):
    '''Takes in ECEF points and a time vector and creates a tracking control.
    '''

    def __init__(self, radar, scan, r=np.linspace(300e3,1000e3,num=10), profiler=None, logger=None, **kwargs):
        super().__init__(radar, profiler=profiler, logger=logger)
        self.scan = scan
        self.r = r

        if self.logger is not None:
            self.logger.info(f'Scanner:init')

    def point_radar(self, t):
        point = self.scan.ecef_pointing(t, self.radar.tx[0])

        point_tx = point + self.radar.tx[0].ecef
        point_rx = point[:,None]*self.r[None,:] + self.radar.tx[0].ecef[:,None]
        
        self.point_tx_ecef(point_tx)
        self.point_rx_ecef(point_rx)

        return self.radar

    def generator(self, t):
        for ti in range(len(t)):
            yield self.point_radar(t[ti])
