#!/usr/bin/env python

'''
'''

import pyant
import numpy as np

from ..radar import Radar
from ..tx_rx import TX, RX


def gen_mock():
    class Omni(pyant.Beam):
        def copy(self):
            ret = Omni(
                azimuth = self.azimuth, 
                elevation = self.elevation,
                frequency = self.frequency,
                radians = self.radians,
            )
            return ret

        def gain(self, k, polarization=None, ind=None):
            if len(k.shape) == 1:
                return 1.0
            else:
                return np.ones((k.shape[1],), dtype=k.dtype)

    tx_beam = Omni(azimuth=0.0, elevation=90.0, frequency=100e6)
    rx_beam = Omni(azimuth=0.0, elevation=90.0, frequency=100e6)

    tx = [TX(
        lat = 90, lon = 0, alt = 0,
        min_elevation = 0,
        beam = tx_beam,
        power = 1.6e6,
        bandwidth = 1e6,
        duty_cycle = 0.25, 
        pulse_length=30.0*64.0*1e-6,
        ipp=20e-3,
        n_ipp=10.0,
    )]
    rx = [RX(
        lat = 90, lon = 0, alt = 0,
        min_elevation = 0,
        noise = 100,
        beam = rx_beam,
    )]
    mock = Radar(
        tx, 
        rx,
        max_off_axis=180.0, 
        min_SNRdb=10.0,
    )
    return mock
    