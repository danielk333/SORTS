#!/usr/bin/env python

'''This module is used to define the radar controller

'''
#Python standard import
from abc import ABC, abstractmethod


#Third party import
import numpy as np

#Local import


class RadarController(ABC):
    '''A radar controller.
    '''

    def __init__(self, radar, t=None, t0=0.0, profiler=None, logger=None):
        self.radar = radar
        self.t = t
        self.t0 = t0
        self.logger = logger
        self.profiler = profiler


    def run(self):
        return self(self.t - self.t0)


    @abstractmethod
    def generator(self, t, **kwargs):
        '''This will configure the radar system and return a pointer to the contained radar system instance with the correct configuration. 
        It should always assume the input `t` is a iterable and use `yield` to return `self.radar`.

        **NOTE:** This is NOT guaranteed to return a copy of the radar system, but rather just a pointer to it.
        '''
        pass


    def __call__(self, t, **kwargs):
        if isinstance(t, float) or isinstance(t, int):
            ret = list(self.generator([t], **kwargs))[0]
        else:
            if len(t) > 1:
                ret = self.generator(t, **kwargs)
            else:
                ret = list(self.generator(t, **kwargs))[0]

        return ret


    @staticmethod
    def _point_station(station, ecef):
        if len(ecef.shape) > 1:
            k = station.point_ecef(ecef - station.ecef[:,None])
        else:
            k = station.point_ecef(ecef - station.ecef)

        #pointing turns on station
        station.enabled = True

        #error check pointing
        keep = k[2,...] >= np.sin(np.radians(station.min_elevation))
        if len(ecef.shape) > 1:
            if not np.any(keep):
                station.enabled = False
            else:
                new_k = k[:,keep]
                station.point(new_k)
        else:
            if not keep:
                station.enabled = False


    def point_rx_ecef(self, ecef):
        '''Point all rx sites into the direction of given ECEF coordinate, relative Earth Center.
        '''
        for rx in self.radar.rx:
            RadarController._point_station(rx, ecef)


    def point_tx_ecef(self, ecef):
        '''Point all tx sites into the direction of given ECEF coordinate, relative Earth Center.
        '''
        for tx in self.radar.tx:
            RadarController._point_station(tx, ecef)


    def point_ecef(self, ecef):
        '''Point all sites into the direction of given ECEF coordinate, relative Earth Center.
        '''
        self.point_tx_ecef(ecef)
        self.point_rx_ecef(ecef)



    def point(self, enu):
        '''Point all sites into the direction of a given East, North, Up (ENU) local coordinate system.
        '''
        for tx in self.radar.tx:
            tx.point(enu)
        for rx in self.radar.rx:
            rx.point(enu)


    def turn_off(self):
        for st in self.radar.tx + self.radar.rx:
            st.enabled = False


    def turn_on(self):
        for st in self.radar.tx + self.radar.rx:
            st.enabled = True