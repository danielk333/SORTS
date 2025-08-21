#!/usr/bin/env python

""" """
# Python standard import


import numpy as np
import pyant

from .radars import radar_generator
from ..radar import Radar
from ..tx_rx import TX, RX


@radar_generator("eiscat_esr", "cassegrain")
def gen_eiscat_esr():
    # TODO: Proper ESR coordinates.  These were eyeballed from maps
    lat_esr32 = 78.153145
    lon_esr32 = 16.0758715
    alt_esr = 185  # TODO: Check!

    # QnD calculations: 42m is 128 m to the East of the 32m
    # (Computing ECEF coordinates from this gives an offset of 128.66 metres)
    baseline = 128
    lon_incr = baseline / (1e7 / 90 * np.cos(np.radians(lat_esr32)))
    lat_esr42 = lat_esr32
    lon_esr42 = lon_esr32 + lon_incr

    tx_beam32 = pyant.beam_of_radar("esr_32m", "cassegrain")
    rx_beam32 = pyant.beam_of_radar("esr_32m", "cassegrain")
    tx_beam42 = pyant.beam_of_radar("esr_42m", "cassegrain")
    rx_beam42 = pyant.beam_of_radar("esr_42m", "cassegrain")

    tx = [
        TX(  # 1st TX: 32m
            lat=lat_esr32,
            lon=lon_esr32,
            alt=alt_esr,
            min_elevation=15,  # TODO: Check
            beam=tx_beam32,
            power=1.0e6,
            bandwidth=2e6,  # Check
            duty_cycle=0.25,
            pulse_length=2e-3,  # semantics?
            ipp=20e-3,
            n_ipp=10,
        ),
        TX(  # 2nd TX: 42m
            lat=lat_esr42,
            lon=lon_esr42,
            alt=alt_esr,
            min_elevation=75,  # TODO: Fix az and el to narrow intervals
            beam=tx_beam42,
            power=1.0e6,
            bandwidth=2e6,  # Check
            duty_cycle=0.25,
            pulse_length=2e-3,  # semantics?
            ipp=20e-3,
            n_ipp=10,
        ),
    ]

    rx = [
        RX(
            lat=lat_esr32,
            lon=lon_esr32,
            alt=alt_esr,
            min_elevation=15,  # TODO: Check
            noise=70,
            beam=rx_beam32,
        ),
        RX(
            lat=lat_esr42,
            lon=lon_esr42,
            alt=alt_esr,
            min_elevation=75,  # TODO: Fix az and el to narrow intervals
            noise=70,
            beam=rx_beam42,
        ),
    ]

    esr = Radar(
        tx,
        rx,
        min_SNRdb=10.0,
        joint_stations=[(0, 0), (1, 1)],
    )
    return esr
