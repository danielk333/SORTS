#!/usr/bin/env python

"""#TODO

"""

import logging
import numpy as np

from .radar_controller import RadarController

logger = logging.getLogger(__name__)

class Scanner(RadarController):
    """Takes in a scan and create a scanning radar controller."""

    META_FIELDS = RadarController.META_FIELDS + [
        "scan_type",
        "dwell",
    ]

    def __init__(
        self,
        radar,
        scan,
        t0=0.0,
        r=np.linspace(300e3, 1000e3, num=10),
        as_altitude=False,
        return_copy=False,
        meta=None,
        **kwargs
    ):
        super().__init__(radar, t0=t0, meta=meta, **kwargs)
        self.scan = scan
        if self.t is not None and self.t_slice is None:
            self.dwell = np.max(self.scan.dwell(self.t))

        self.r = r
        self.return_copy = return_copy
        self.as_altitude = as_altitude

        logger.info(f"Scanner:init")

    @property
    def dwell(self):
        if self.t_slice is None:
            return self.scan.dwell(self.t)
        else:
            return self.t_slice

    @dwell.setter
    def dwell(self, val):
        self.t_slice = val

    def default_meta(self):
        dic = super().default_meta()
        dic["scan_type"] = self.scan.__class__
        return dic

    def point_radar(self, t):
        """Assumes t is not array"""

        if self.return_copy:
            radar = self.radar.copy()
        else:
            radar = self.radar

        meta = self.default_meta()
        meta["dwell"] = self.scan.dwell(t)

        RadarController.coh_integration(radar, meta["dwell"])

        point_rx_to_tx = []
        point_tx = []
        for tx in radar.tx:
            point = self.scan.ecef_pointing(t, tx)

            if self.as_altitude:
                if len(point.shape) > 1:
                    r = self.r[None, :] / point[2, :]
                    point_tx.append(point + tx.ecef[:, None])
                    __ptx = point[:, :, None] * r[None, :, :] + tx.ecef[:, None, None]
                    point_rx_to_tx.append(__ptx.reshape(3, __ptx.shape[1] * __ptx.shape[2]))
                else:
                    r = self.r / point[2]
                    point_tx.append(point + tx.ecef)
                    point_rx_to_tx.append(point[:, None] * r[None, :] + tx.ecef[:, None])
            else:
                if len(point.shape) > 1:
                    point_tx.append(point + tx.ecef[:, None])
                    __ptx = point[:, :, None] * self.r[None, None, :] + tx.ecef[:, None, None]
                    point_rx_to_tx.append(__ptx.reshape(3, __ptx.shape[1] * __ptx.shape[2]))
                else:
                    point_tx.append(point + tx.ecef)
                    point_rx_to_tx.append(point[:, None] * self.r[None, :] + tx.ecef[:, None])

            RadarController._point_station(tx, point_tx[-1])

        for rx in radar.rx:
            rx_point = []
            for txi, tx in enumerate(radar.tx):
                # < 200 meters apart = same location for pointing
                if np.linalg.norm(tx.ecef - rx.ecef) < 200.0:
                    __ptx = point_tx[txi]
                    if len(__ptx.shape) == 1:
                        __ptx = __ptx.reshape(3, 1)
                    rx_point.append(__ptx)
                else:
                    rx_point.append(point_rx_to_tx[txi])
            rx_point = np.concatenate(rx_point, axis=1)

            if len(rx_point.shape) > 1 and rx_point.size == 3:
                rx_point.shape = (3,)

            RadarController._point_station(rx, rx_point)

        # Make sure radar is on
        self.toggle_stations(t, radar)

        return radar, meta

    def generator(self, t):
        for ti in range(len(t)):
            yield self.point_radar(t[ti])
