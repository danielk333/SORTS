#!/usr/bin/env python

"""

"""

import numpy as np

from .scan import Scan


class Plane(Scan):
    """A uniform sampling of a horizontal plane."""

    def __init__(
        self,
        min_elevation=30.0,
        altitude=200e3,
        x_size=50e3,
        y_size=50e3,
        x_num=20,
        y_num=20,
        dwell=0.1,
        x_offset=0.0,
        y_offset=0.0,
    ):
        super().__init__(coordinates="enu")
        self._dwell = dwell
        self.min_elevation = min_elevation
        self.altitude = altitude

        self.x_size = x_size
        self.y_size = y_size
        self.x_num = x_num
        self.y_num = y_num

        k = np.empty((3, x_num * y_num), dtype=np.float64)

        xv, yv = np.meshgrid(
            np.linspace(-x_size * 0.5, x_size * 0.5, num=self.x_num, endpoint=True) + x_offset,
            np.linspace(-y_size * 0.5, y_size * 0.5, num=self.y_num, endpoint=True) + y_offset,
            sparse=False,
            indexing="ij",
        )

        k[0, :] = xv.flatten()
        k[1, :] = yv.flatten()
        k[2, :] = altitude

        k = k / np.linalg.norm(k, axis=0)

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
