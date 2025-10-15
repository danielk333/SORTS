#!/usr/bin/env python

""" """
import numpy as np
import pyant

from .radars import radar_generator
from ..radar import Radar
from ..tx_rx import TX, RX


def gen_nostra_beam():
    beam = pyant.models.Airy(
        pointing=np.array([0, 0, 1], dtype=np.float64),
        frequency=930e6,
        radius=23.0,
        peak_gain=10**4.81,
    )
    return beam


@radar_generator("nostra", "example1")
def gen_nostra():
    """The NOSTRA system."""
    dwell_time = 0.1
    tx_kw = dict(
        power=500e3,
        bandwidth=100e3,
        duty_cycle=0.25,
        pulse_length=1920e-6,
        ipp=10e-3,
        n_ipp=int(dwell_time / 10e-3),
        min_elevation=30.0,
    )
    rx_kw = dict(
        noise=300,
        min_elevation=30.0,
    )

    se_rx = RX(lat=65.89, lon=20.18, alt=0, beam=gen_nostra_beam(), **rx_kw)
    se_tx = TX(lat=65.89, lon=20.18, alt=0, beam=gen_nostra_beam(), **tx_kw)

    no_rx = RX(lat=68.96, lon=18.135, alt=0, beam=gen_nostra_beam(), **rx_kw)
    no_tx = TX(lat=68.96, lon=18.135, alt=0, beam=gen_nostra_beam(), **tx_kw)

    fi_rx = RX(lat=67.80, lon=27.684, alt=0, beam=gen_nostra_beam(), **rx_kw)
    fi_tx = TX(lat=67.80, lon=27.684, alt=0, beam=gen_nostra_beam(), **tx_kw)
    # define transmit and receive antennas for a radar network.
    tx = [se_tx, no_tx, fi_tx]
    rx = [se_rx, no_rx, fi_rx]

    nostra = Radar(
        tx=tx,
        rx=rx,
        min_SNRdb=12.0,
        joint_stations=[(0, 0), (1, 1), (2, 2)],
    )
    return nostra
