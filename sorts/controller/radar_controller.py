#!/usr/bin/env python

'''This module is used to define the radar controller

'''
#Python standard import
from abc import ABC, abstractmethod
import copy

#Third party import
import numpy as np

#Local import


class RadarController(ABC):
    '''A radar controller.
    '''

    META_FIELDS = [
        'controller_type',
    ]

    def __init__(self, radar, t=None, t0=0.0, t_slice=None, profiler=None, logger=None, meta=None):
        self.radar = radar
        self.t = t
        self.t0 = t0
        self.t_slice = t_slice
        self.logger = logger
        self.profiler = profiler

        self.meta = dict()
        if meta is not None:
            self.meta.update(meta)
        for key in self.META_FIELDS:
            if key not in self.meta:
                self.meta[key] = None


    def run(self):
        return self(self.t - self.t0)


    def default_meta(self):
        '''This is used to generate meta data on the fly, rather then the static data that can be set in the `self.meta`.
        '''
        meta = dict()
        meta.update(self.meta)
        meta['controller_type'] = self.__class__
        return meta


    @abstractmethod
    def generator(self, t, **kwargs):
        '''This will configure the radar system and return a pointer to the contained radar system instance with the correct configuration. 
        It should always assume the input `t` is an iterable and use `yield` to return `radar, meta`. The `meta` variable should be 
        a dict with the fields defined in `META_FIELDS`

        **NOTE:** This is NOT guaranteed to return a copy of the radar system, however, the subclass should implement this as a option.
        '''
        pass


    def __call__(self, t, **kwargs):
        if isinstance(t, float) or isinstance(t, int):
            ret = list(self.generator([t], **kwargs))[0]
        else:
            if len(t) > 0:
                ret = self.generator(t, **kwargs)
            else:
                ret = []

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

    @staticmethod
    def point_rx_ecef(radar, ecef):
        '''Point all rx sites into the direction of given ECEF coordinate, relative Earth Center.
        '''
        for rx in radar.rx:
            RadarController._point_station(rx, ecef)

    @staticmethod
    def point_tx_ecef(radar, ecef):
        '''Point all tx sites into the direction of given ECEF coordinate, relative Earth Center.
        '''
        for tx in radar.tx:
            RadarController._point_station(tx, ecef)

    @staticmethod
    def point_ecef(radar, ecef):
        '''Point all sites into the direction of given ECEF coordinate, relative Earth Center.
        '''
        RadarController.point_tx_ecef(radar, ecef)
        RadarController.point_rx_ecef(radar, ecef)


    @staticmethod
    def point(radar, enu):
        '''Point all sites into the direction of a given East, North, Up (ENU) local coordinate system.
        '''
        for tx in radar.tx:
            tx.point(enu)
        for rx in radar.rx:
            rx.point(enu)

    @staticmethod
    def turn_off(radar):
        for st in radar.tx + radar.rx:
            st.enabled = False

    @staticmethod
    def turn_on(radar):
        for st in radar.tx + radar.rx:
            st.enabled = True