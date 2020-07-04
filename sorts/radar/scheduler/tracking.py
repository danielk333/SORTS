#!/usr/bin/env python

'''

'''
from abc import abstractmethod

import numpy as np
import pyorb

from ..controller import Tracker
from .scheduler import Scheduler


class Tracking(Scheduler):
    '''
    '''

    def __init__(self, radar, propagator, mjd0, timeslice, allocation, orbit_kw = {}):
        super().__init__(radar)
        self.propagator = propagator
        self.mjd0 = mjd0
        
        self.timeslice = timeslice
        self.allocation = allocation

        if 'M0' not in orbit_kw:
            orbit_kw['M0'] = pyorb.M_earth
        self.orbits = pyorb.Orbit(
            direct_update=True, 
            auto_update=True, 
            num=0,
            **orbit_kw
        )
        self.object_params = []


    @abstractmethod
    def get_priority(self):
        pass


    def update(self, mjd0, epochs, num=0, orbits=None, params=None):
        self.mjd0 = mjd0
        self.orbits.epoch = np.append(self.orbits.epoch, epochs)
        if num > 0:
            self.orbits.add(num=num, **orbits)
            self.object_params += params


    def get_controllers(self):

        self.measurements = []

        #simulate_tracking here

        ctrls = []
        for ind in range(len(self.orbits)):
            t = self.measurements[ind] 
            states = self.propagator.propagate(t, self.orbits.cartesian[:,ind], self.orbits.epoch[ind], **self.object_params[ind])

            ctrl = Tracker(radar = self.radar, t=tv[ind], ecefs = states[:3,:])
            ctrls.append(ctrl)
        
        return ctrls


