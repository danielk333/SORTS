#!/usr/bin/env python

'''
'''

import pyant

from ..radar import Radar
from ..station import TX, RX
from ....transformations import frames


def gen_eiscat_uhf():

    lat_tro = 69.0 + frames.arctime_to_degrees(35, 11)
    lon_tro = 19.0 + frames.arctime_to_degrees(13, 38)
    alt_tro = 86.0

    lat_krn = 67.0 + frames.arctime_to_degrees(51, 38)
    lon_krn = 20.0 + frames.arctime_to_degrees(26, 7)
    alt_krn = 418.0

    lat_sod = 67.0 + frames.arctime_to_degrees(21, 49)
    lon_sod = 26.0 + frames.arctime_to_degrees(37, 37)
    alt_sod = 197.0

    tx_beam = pyant.beam_of_radar('eiscat_uhf', 'measured')
    rx_beam = pyant.beam_of_radar('eiscat_uhf', 'measured')

    tx = [TX(
        lat=lat_tro,
        lon=lon_tro,
        alt=alt_tro,
        min_elevation=30,
        beam=tx_beam,
        power=1.6e6,
        bandwidth=1e6,
        duty_cycle=0.125,
        pulse_length=30.0*64.0*1e-6,
        ipp=20e-3,
        n_ipp=10.0,
    )]

    rx = []

    rx += [RX(
        lat=lat_tro,
        lon=lon_tro,
        alt=alt_tro,
        min_elevation=30,
        noise_temperature=100,
        beam=rx_beam.copy(),
    )]
    rx += [RX(
        lat=lat_krn,
        lon=lon_krn,
        alt=alt_krn,
        min_elevation=30,
        noise_temperature=100,
        beam=rx_beam.copy(),
    )]
    rx += [RX(
        lat=lat_sod,
        lon=lon_sod,
        alt=alt_sod,
        min_elevation=30,
        noise_temperature=100,
        beam=rx_beam.copy(),
    )]

    uhf = Radar(
        tx,
        rx,
        min_SNRdb=10.0,
        joint_stations=[(0, 0)],
    )
    return uhf
