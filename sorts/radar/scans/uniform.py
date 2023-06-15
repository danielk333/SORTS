#!/usr/bin/env python

"""

"""

import numpy as np

from .scan import Scan


class Uniform(Scan):
    """Uniformly sample the FOV using Fibonacci lattice."""

    def __init__(self, min_elevation=30.0, dwell=0.1, sph_num=1000):
        super().__init__(coordinates="enu")
        self._dwell = dwell
        self.min_elevation = min_elevation
        self.num = sph_num
        self.sph_num = sph_num

        k = np.empty((3, self.num), dtype=np.float64)

        golden_ratio = (1 + 5**0.5) / 2
        inds = np.arange(0, self.num)
        theta = 2 * np.pi * inds / golden_ratio
        phi = np.arccos(1 - 2 * (inds + 0.5) / self.num)

        k[0, :] = np.cos(theta) * np.sin(phi)
        k[1, :] = np.sin(theta) * np.sin(phi)
        k[2, :] = np.cos(phi)

        min_z = np.sin(np.radians(min_elevation))
        k = k[:, k[2, :] >= min_z]
        self.num = k.shape[1]
        self.pointings = k

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
        return self.pointings[:, ind]
