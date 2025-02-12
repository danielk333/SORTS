#!/usr/bin/env python

"""

"""

import numpy as np

from .scan import Scan


class Fence(Scan):
    """General fence scan."""

    def __init__(self, azimuth=0.0, min_elevation=30.0, dwell=0.2, num=20):
        super().__init__(coordinates="azelr")
        self._dwell = dwell
        self.num = num
        self.min_elevation = min_elevation
        self.azimuth = azimuth

        self._az = np.empty((num,), dtype=np.float64)
        self._el = np.linspace(min_elevation, 180 - min_elevation, num=num, dtype=np.float64)
        inds_ = self._el > 90.0
        self._az[inds_] = np.mod(self.azimuth + 180.0, 360.0)
        self._az[np.logical_not(inds_)] = self.azimuth

        self._el[inds_] = 180.0 - self._el[inds_]

    def dwell(self, t=None):
        if t is None:
            return self._dwell
        else:
            if isinstance(t, float) or isinstance(t, int):
                return self._dwell
            else:
                return np.ones(t.shape, dtype=t.dtype) * self._dwell

    def min_dwell(self):
        return self._dwell

    def cycle(self):
        return self.num * self._dwell

    def pointing(self, t):
        ind = (np.mod(t / self.cycle(), 1) * self.num).astype(np.int64)
        if isinstance(t, float) or isinstance(t, int):
            shape = (3,)
        else:
            shape = (3, len(t))

        azelr = np.empty(shape, dtype=np.float64)
        azelr[0, ...] = self._az[ind]
        azelr[1, ...] = self._el[ind]
        azelr[2, ...] = 1.0
        return azelr
