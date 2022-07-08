#!/usr/bin/env python

'''
'''

import pyant

from ..radar import Radar
from ..station import TX, RX


def gen_tromso_space_debris_radar(fence=False, phased=False):
    lat = 69.5866115
    lon = 19.221555
    alt = 85.0

    if phased:
        rx_beam = pyant.beam_of_radar('tsdr', 'phased_fcp')
        tx_beam = pyant.beam_of_radar('tsdr', 'phased_fcp')
    else:
        rx_beam = pyant.beam_of_radar('tsdr', 'fcp')
        tx_beam = pyant.beam_of_radar('tsdr', 'fcp')

    if fence:
        rx_beam.sph_point(azimuth=0.0, elevation=[30.0, 60.0, 90.0, 100.0])
        rx_beam.width = rx_beam.width/4

        tx_beam.sph_point(azimuth=0.0, elevation=[30.0, 60.0, 90.0, 100.0])
        tx_beam.width = tx_beam.width/4
    else:
        rx_beam.sph_point(azimuth=0.0, elevation=90.0)
        tx_beam.sph_point(azimuth=0.0, elevation=90.0)

    tsr_tx = TX(
        lat=lat,
        lon=lon,
        alt=alt,
        min_elevation=0,
        beam=tx_beam,
        power=500.0e3,
        bandwidth=1e7,
        duty_cycle=0.125,
        pulse_length=0.0002,
        ipp=10e-3,
        n_ipp=10,
    )

    tsr_rx = RX(
        lat=lat,
        lon=lon,
        alt=alt,
        min_elevation=0,
        noise_temperature=100,
        beam=rx_beam,
    )

    tx = [tsr_tx]
    rx = [tsr_rx]

    tsdr_r = Radar(
        tx,
        rx,
        min_SNRdb=10.0,
        joint_stations=[(0, 0)],
    )
    return tsdr_r
