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

    def __init__(self, radar):
        self.radar = radar


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


    def point_ecef(self, ecef):
        '''Point all sites into the direction of given ECEF coordinate, relative Earth Center.
        '''
        for tx in self.radar.tx:
            tx.point_ecef(ecef - tx.ecef)
        for rx in self.radar.rx:
            rx.point_ecef(ecef - rx.ecef)


    def point(self, enu):
        '''Point all sites into the direction of a given East, North, Up (ENU) local coordinate system.
        '''
        for tx in self.radar.tx:
            tx.point(enu)
        for rx in self.radar.rx:
            rx.point(enu)

