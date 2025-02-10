#!/usr/bin/env python

""" """
import pyant

from .radars import radar_generator
from ..radar import Radar
from ..tx_rx import TX, RX


def gen_nostra_beam():
    beam = pyant.models.Gaussian(
        azimuth=0,
        elevation=90.0,
        frequency=3.4e9,
        I0=10**5.81,
        radius=10.0,
        normal_azimuth=0,
        normal_elevation=90.0,
        degrees=True,
    )
    return beam


@radar_generator("nostra", "example1")
def gen_nostra():
    """The NOSTRA system."""
    dwell_time = 0.1
    tx_kw = dict(
        power=5e6,
        bandwidth=100e3,
        duty_cycle=0.25,
        pulse_length=1920e-6,
        ipp=10e-3,
        n_ipp=int(dwell_time / 10e-3),
        min_elevation=45.0,
    )
    rx_kw = dict(
        noise=150,
        min_elevation=45.0,
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
        min_SNRdb=10.0,
        joint_stations=[(0, 0), (1, 1), (2, 2)],
    )
    return nostra
