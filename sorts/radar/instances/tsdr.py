#!/usr/bin/env python

'''
'''
#Python standard import
import pkg_resources

import numpy as np
import pyant.instances as alib
import pyant

from ..radar import Radar
from ..tx_rx import TX, RX


def gen_tromso_space_debris_radar(fence=False):
    lat = 69.5866115
    lon = 19.221555 
    alt = 85.0

    if fence:
        rx_beam = alib.tsdr_fence.copy()
        tx_beam = alib.tsdr_fence.copy()
    else:
        rx_beam = alib.tsdr.copy()
        tx_beam = alib.tsdr.copy()

    tsr_tx = antenna.AntennaTX(
        lat = lat,
        lon = lon,
        alt = alt,
        min_elevation = 30.0,
        beam = tx_beam,
        power = 500.0e3,
        bandwidth = 1e6,
        duty_cycle = 0.125, 
        pulse_length=30.0*64.0*1e-6,
        ipp=20e-3,
        n_ipp=10.0,
    )

    tsr_rx = RX(
        lat = lat,
        lon = lon,
        alt = alt,
        min_elevation = 30.0,
        noise = 100,
        beam = rx_beam,
    )

    tx=[tsr_tx]
    rx=[tsr_rx]

    tsdr_r = Radar(tx, rx
        max_off_axis=120.0, 
        min_SNRdb=10.0,
    )
    return tsdr_r
    
    