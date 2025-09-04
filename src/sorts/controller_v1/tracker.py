#!/usr/bin/env python

"""#TODO

"""

import logging
import numpy as np

from .radar_controller import RadarController

logger = logging.getLogger(__name__)

class Tracker(RadarController):
    """Takes in ECEF points and a time vector and creates a tracking control."""

    META_FIELDS = RadarController.META_FIELDS + [
        "dwell",
        "target",
    ]

    def __init__(
        self,
        radar,
        t,
        ecefs,
        t0=0.0,
        dwell=0.1,
        return_copy=True,
        meta=None,
        **kwargs
    ):
        super().__init__(radar, t=t, t0=t0, meta=meta, **kwargs)
        self.ecefs = ecefs
        self.dwell = dwell
        self.return_copy = return_copy

        logger.info(f"Tracker:init")

    @property
    def dwell(self):
        return self.t_slice

    @dwell.setter
    def dwell(self, val):
        self.t_slice = val

    def default_meta(self):
        dic = super().default_meta()
        dic["dwell"] = self.dwell
        return dic

    def point_radar(self, radar, ind):
        RadarController.point_tx_ecef(radar, self.ecefs[:3, ind])
        RadarController.point_rx_ecef(radar, self.ecefs[:3, ind])

    def generator(self, t):
        logger.debug(f"Tracker:generator: len(t) = {len(t)}")

        for ti in range(len(t)):
            if self.return_copy:
                radar = self.radar.copy()
            else:
                radar = self.radar

            dt = t[ti] - self.t
            check = np.logical_and(dt >= 0, dt < self.dwell)
            ind = np.argmax(check)
            meta = self.default_meta()

            RadarController.coh_integration(radar, self.dwell)

            self.toggle_stations(t[ti], radar)
            self.point_radar(radar, ind)

            yield radar, meta

        logger.debug(f"Tracker:generator:completed")
