#!/usr/bin/env python

'''

'''

import numpy as np

from .scan import Scan

class Beampark(Scan):
    '''General fence scan.
    '''
    def __init__(self, azimuth = 0.0, elevation=90.0):
        super().__init__(coordinates='azelr')
        self.elevation = elevation
        self.azimuth = azimuth


    def dwell(self, t=None):
        return None

    def min_dwell(self):
        return None

    def cycle(self):
        return None

    def pointing(self, t):
        if isinstance(t, float) or isinstance(t, int):
            shape = (3, )
        else:
            shape = (3, len(t))

        azelr = np.empty(shape, dtype=np.float64)
        azelr[0,...] = self.elevation
        azelr[1,...] = self.azimuth
        azelr[2,...] = 1.0
        return azelr