#!/usr/bin/env python

'''

'''

import numpy as np

from .scan import Scan


class Beampark(Scan):
    '''Static beampark.
    '''

    def __init__(self, azimuth=0.0, elevation=90.0, dwell=0.1):
        super().__init__(coordinates='azelr')
        self.elevation = elevation
        self.azimuth = azimuth
        self._dwell = dwell

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
        return None

    def pointing(self, t):
        if isinstance(t, float) or isinstance(t, int):
            shape = (3, )
        else:
            shape = (3, len(t))

        el_len = True
        try:
            shape += (len(self.elevation), )
        except TypeError:
            el_len = False

        az_len = True
        try:
            shape += (len(self.azimuth), )
        except TypeError:
            az_len = False

        azelr = np.empty(shape, dtype=np.float64)

        if az_len:
            for ind in range(len(self.azimuth)):
                if len(shape) == 2:
                    azelr[0, ind] = self.azimuth[ind]
                else:
                    azelr[0, :, ind] = self.azimuth[ind]
        else:
            azelr[0, ...] = self.azimuth

        if el_len:
            for ind in range(len(self.elevation)):
                if len(shape) == 2:
                    azelr[1, ind] = self.elevation[ind]
                else:
                    azelr[1, :, ind] = self.elevation[ind]
        else:
            azelr[1, ...] = self.elevation

        azelr[2, ...] = 1.0
        return azelr
