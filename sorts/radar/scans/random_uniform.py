#!/usr/bin/env python

'''

'''

import numpy as np

from .scan import Scan

class RandomUniform(Scan):
    '''Uniform randomly distributed points in the FOV.
    '''
    def __init__(self, min_elevation=30.0, dwell=0.1, cycle_num = 10000):
        super().__init__(coordinates='enu')
        self._dwell = dwell
        self.num = cycle_num
        self.min_elevation = min_elevation

        min_z = np.sin(np.radians(min_elevation))

        theta = 2*np.pi*np.random.rand(self.num)
        phi = np.arccos(np.random.rand(self.num)*(1 - min_z) + min_z)

        k = np.empty((3, self.num), dtype=np.float64)

        k[0,:] = np.cos(theta)*np.sin(phi)
        k[1,:] = np.sin(theta)*np.sin(phi)
        k[2,:] = np.cos(phi)

        self.pointings = k


    def dwell(self, t=None):
        if t is None:
            return self._dwell
        else:
            if isinstance(t, float) or isinstance(t, int):
                return self._dwell
            else:
                return np.ones(t.shape, dtype=t.dtype)*self._dwell


    def min_dwell(self):
        return self._dwell


    def cycle(self):
        return self.num*self._dwell


    def pointing(self, t):
        ind = (np.mod(t/self.cycle(), 1)*self.num).astype(np.int)
        return self.pointings[:,ind]