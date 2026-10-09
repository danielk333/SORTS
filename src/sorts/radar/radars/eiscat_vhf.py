#!/usr/bin/env python

""" """

import typing as t
from radardef.types import EiscatUHFLocation
from radardef.radar_stations import EiscatUHF
from .radars import radar_generator
from ..radar import Radar
from ..tx_rx import TX, RX


# TODO: we should dissolve the `Radar` and `Station` classes and use `RadarStation` from `radardef`?
@radar_generator("eiscat_uhf", "measured")
def gen_eiscat_uhf():
    # NOTE: `cast` is needed because for some reason `EiscatUHFLocation.TROMSO` etc is inferred to be type `int`
    tro_stn = EiscatUHF(t.cast(EiscatUHFLocation, EiscatUHFLocation.TROMSO))
    krn_stn = EiscatUHF(t.cast(EiscatUHFLocation, EiscatUHFLocation.KIRUNA))
    sod_stn = EiscatUHF(t.cast(EiscatUHFLocation, EiscatUHFLocation.SODANKYLA))

    tx = [
        TX(
            lat=tro_stn.lat,
            lon=tro_stn.lon,
            alt=tro_stn.alt,
            min_elevation=tro_stn.min_elevation,
            beam=tro_stn.beam,
            beam_parameters=tro_stn.beam_parameters,
            power=tro_stn.power,
            bandwidth=1e6,
            duty_cycle=0.125,
            pulse_length=30.0 * 64.0 * 1e-6,
            ipp=20e-3,
            n_ipp=10.0,
        )
    ]

    rx = []

    rx += [
        RX(
            lat=tro_stn.lat,
            lon=tro_stn.lon,
            alt=tro_stn.alt,
            min_elevation=tro_stn.min_elevation,
            noise=100,
            beam=tro_stn.beam,
            beam_parameters=tro_stn.beam_parameters,
        )
    ]
    rx += [
        RX(
            lat=krn_stn.lat,
            lon=krn_stn.lon,
            alt=krn_stn.alt,
            min_elevation=krn_stn.min_elevation,
            noise=100,
            beam=krn_stn.beam,
            beam_parameters=krn_stn.beam_parameters,
        )
    ]
    rx += [
        RX(
            lat=sod_stn.lat,
            lon=sod_stn.lon,
            alt=sod_stn.alt,
            min_elevation=sod_stn.min_elevation,
            noise=100,
            beam=sod_stn.beam,
            beam_parameters=sod_stn.beam_parameters,
        )
    ]

    uhf = Radar(
        tx,
        rx,
        min_SNRdb=10.0,
        joint_stations=[(0, 0)],
    )
    return uhf
