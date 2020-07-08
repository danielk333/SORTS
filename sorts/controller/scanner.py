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
        point_rx_to_tx = []
        point_tx = []
        for tx in self.radar.tx:
            point = self.scan.ecef_pointing(t, tx)

            point_tx.append(point + tx.ecef)
            point_rx_to_tx.append(point[:,None]*self.r[None,:] + tx.ecef[:,None])
            
            RadarController._point_station(tx, point_tx[-1])

        for rx in self.radar.rx:
            rx_point = []
            for txi, tx in enumerate(self.radar.tx):
                #< 200 meters apart = same location for pointing
                if np.linalg.norm(tx.ecef - rx.ecef) < 200.0:
                    rx_point.append(point_tx[txi].reshape(3,1))
                else:
                    rx_point.append(point_rx_to_tx[txi])
            rx_point = np.concatenate(rx_point, axis=1)

            RadarController._point_station(rx, rx_point)

        return self.radar

    def generator(self, t):
        for ti in range(len(t)):
            yield self.point_radar(t[ti])
