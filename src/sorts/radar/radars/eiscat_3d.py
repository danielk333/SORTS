#!/usr/bin/env python

"""
The EISCAT 3D system.

The EISCAT 3D system in stage 1.

For more information see:
    * `EISCAT <https://eiscat.se/>`_
    * `EISCAT 3D <https://www.eiscat.se/eiscat3d/>`_


# TODO update these

**EISCAT 3D Stages:**

    * Stage 1: x
    * Stage 2: x
    * Stage 3: x



"""
import pyant

from .radars import radar_generator
from ..radar import Radar
from ..tx_rx import TX, RX


SKI_LAT = 69.34023844
SKI_LON = 20.313166
SKI_ALT = 0.0

KAR_LAT = 68.463862
KAR_LON = 22.458859
KAR_ALT = 0.0

KAI_LAT = 68.148205
KAI_LON = 19.769894
KAI_ALT = 0.0

KIR_LAT = 67.860308
KIR_LON = 20.432841
KIR_ALT = 300.0


def gen_eiscat3d_stage1(beam_type, path=None, **interp_kw):
    """ """
    if beam_type == "array":
        tx_beam_ski = pyant.beam_of_radar("e3d_stage1", "array")
        rx_beam_ski = pyant.beam_of_radar("e3d_stage2", "array")
        rx_beam_kar = pyant.beam_of_radar("e3d_stage1", "array")
        rx_beam_kai = pyant.beam_of_radar("e3d_stage1", "array")
    if beam_type == "interp":
        tx_beam_ski = pyant.beam_of_radar("e3d_stage1", "interpolated_array", **interp_kw)
        rx_beam_ski = pyant.beam_of_radar("e3d_stage2", "interpolated_array", **interp_kw)
        rx_beam_kar = pyant.beam_of_radar("e3d_stage1", "interpolated_array", **interp_kw)
        rx_beam_kai = pyant.beam_of_radar("e3d_stage1", "interpolated_array", **interp_kw)
    rx_kw = dict(
        min_elevation=30.0,
        noise=150,
    )
    ski = RX(lat=SKI_LAT, lon=SKI_LON, alt=SKI_ALT, beam=rx_beam_ski, **rx_kw)
    dwell_time = 0.1
    ski_tx = TX(
        lat=SKI_LAT,
        lon=SKI_LON,
        alt=SKI_ALT,
        min_elevation=30.0,
        beam=tx_beam_ski,
        power=5e6,  # 5 MW
        bandwidth=100e3,  # 100 kHz tx bandwidth
        duty_cycle=0.25,  # 25% duty-cycle
        pulse_length=1920e-6,
        ipp=10e-3,
        n_ipp=int(dwell_time / 10e-3),
    )
    kar = RX(lat=KAR_LAT, lon=KAR_LON, alt=KAR_ALT, beam=rx_beam_kar, **rx_kw)
    kai = RX(lat=KAI_LAT, lon=KAI_LON, alt=KAI_ALT, beam=rx_beam_kai, **rx_kw)
    # define transmit and receive antennas for a radar network.
    tx = [ski_tx]
    rx = [ski, kar, kai]

    eiscat3d = Radar(
        tx=tx,
        rx=rx,
        min_SNRdb=10.0,
        joint_stations=[(0, 0)],
    )
    return eiscat3d


@radar_generator("eiscat3d", "stage1-array")
def gen_eiscat3d_stage1_array(path=None):
    return gen_eiscat3d_stage1("array", path=None)


@radar_generator("eiscat3d", "stage1-interp")
def gen_eiscat3d_stage1_interp(path=None, **interpolation_kwargs):
    return gen_eiscat3d_stage1("interp", path=None, **interpolation_kwargs)


@radar_generator("eiscat3d", "demo-array")
def gen_eiscat3d_demonstrator():
    """The EISCAT 3D demonstrator module."""
    tx_beam_kir = pyant.beam_of_radar("e3d_module", "array")
    rx_beam_kir = pyant.beam_of_radar("e3d_module", "array")

    kir = RX(
        lat=KIR_LAT,
        lon=KIR_LON,
        alt=KIR_ALT,
        min_elevation=30.0,
        noise=150,
        beam=rx_beam_kir,
    )
    dwell_time = 0.1
    kir_tx = TX(
        lat=KIR_LAT,
        lon=KIR_LON,
        alt=KIR_ALT,
        min_elevation=30.0,
        beam=tx_beam_kir,
        power=0.5e3 * 2 * 16,  # 500W / antenna / pol (now 16 antennas)
        bandwidth=100e3,  # 100 kHz tx bandwidth
        duty_cycle=0.25,  # 25% duty-cycle
        pulse_length=1920e-6,
        ipp=10e-3,
        n_ipp=int(dwell_time / 10e-3),
    )

    # define transmit and receive antennas for a radar network.
    tx = [kir_tx]
    rx = [kir]

    eiscat3d_demonstartor = Radar(
        tx=tx,
        rx=rx,
        min_SNRdb=10.0,
        joint_stations=[(0, 0)],
    )
    return eiscat3d_demonstartor
