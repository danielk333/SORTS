#!/usr/bin/env python

'''#TODO

'''

import numpy as np

from .radar_controller import RadarController
from ..radar.scans import Beampark


class Static(RadarController):
    '''Takes in a direction and creates a static radar in beam park mode.
    '''

    META_FIELDS = RadarController.META_FIELDS + [
        'scan_type',
        'dwell',
    ]

    def __init__(self, radar, azimuth=0.0, elevation=90.0, r=np.linspace(300e3,1000e3,num=10), profiler=None, logger=None, meta=None, **kwargs):
        super().__init__(radar.copy(), profiler=profiler, logger=logger, meta=meta, **kwargs)
        if self.meta['dwell'] is None:
            self.meta['dwell'] = 0.1

        self.scan = Beampark(azimuth = azimuth, elevation=elevation, dwell = self.meta['dwell'])
        self.r = r

        if self.logger is not None:
            self.logger.info(f'Static:init')

        self.point_radar()
        self.turn_on(self.radar)

    def default_meta(self):
        dic = super().default_meta()
        dic['scan_type'] = self.scan.__class__
        return dic

    def point_radar(self):
        '''Assumes t is not array
        '''
        if self.profiler is not None:
            self.profiler.start('Static:generator:point_radar')

        t = 0.0
        point_rx_to_tx = []
        point_tx = []
        for tx in self.radar.tx:
            point = self.scan.ecef_pointing(t, tx)

            if len(point.shape) > 1:
                point_tx.append(point + tx.ecef[:,None])
                __ptx = point[:,:,None]*self.r[None,None,:] + tx.ecef[:,None,None]
                point_rx_to_tx.append(__ptx.reshape(3, __ptx.shape[1]*__ptx.shape[2]))
            else:
                point_tx.append(point + tx.ecef)
                point_rx_to_tx.append(point[:,None]*self.r[None,:] + tx.ecef[:,None])
            
            RadarController._point_station(tx, point_tx[-1])

        for rx in self.radar.rx:
            rx_point = []
            for txi, tx in enumerate(self.radar.tx):
                #< 200 meters apart = same location for pointing
                if np.linalg.norm(tx.ecef - rx.ecef) < 200.0:
                    __ptx = point_tx[txi]
                    if len(__ptx.shape) == 1:
                        __ptx = __ptx.reshape(3,1)
                    rx_point.append(__ptx)
                else:
                    rx_point.append(point_rx_to_tx[txi])
            rx_point = np.concatenate(rx_point, axis=1)

            RadarController._point_station(rx, rx_point)

        RadarController.coh_integration(radar, self.meta['dwell'])

        if self.profiler is not None:
            self.profiler.stop('Static:generator:point_radar')

    def generator(self, t):
        for ti in range(len(t)):
            yield self.radar, self.meta
