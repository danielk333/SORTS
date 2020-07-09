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


def eiscat3d_interp(tx_fnames=None, rx_fnames=None, res=500):
    
    tx_intp = []
    for txi in range(1):
        tx_intp += [pyant.PlaneArrayInterp(
            azimuth=0,
            elevation=90, 
            frequency=233e6,
        )]

        if tx_fnames is None:
            stream = pkg_resources.resource_stream('sorts.data', f'e3d_tx{txi}_res{res}_interp.npy')
            tx_intp[-1].load(stream)
        else:    
            tx_intp[-1].load(tx_fnames[txi])
    
    rx_intp = []
    for rxi in range(3):
        rx_intp += [pyant.PlaneArrayInterp(
            azimuth=0,
            elevation=90, 
            frequency=233e6,
        )]

        if rx_fnames is None:
            stream = pkg_resources.resource_stream('sorts.data', f'e3d_rx{rxi}_res{res}_interp.npy')
            rx_intp[-1].load(stream)
        else:    
            rx_intp[-1].load(rx_fnames[rxi])
        
    return tx_intp, rx_intp


def gen_eiscat3d(beam='array', stage=1):
    '''The EISCAT 3D system.

    :param str beam: Decides what initial antenna radiation-model to use.
    :param int stage: The stage of development of EISCAT 3D. 

    The EISCAT 3D system in stage 1.

    For more information see:
      * `EISCAT <https://eiscat.se/>`_
      * `EISCAT 3D <https://www.eiscat.se/eiscat3d/>`_


    **EISCAT 3D Stages:**

      * Stage 1: As of writing it is assumed to have all of the antennas in place but only transmitters on half of the antennas in a dense core ,i.e. TX will have 42 dB peak gain while RX still has 45 dB peak gain. 3 Sites will exist, one is a TX and RX, the other 2 RX sites.
      * Stage 2: Both TX and RX sites will have 45 dB peak gain.
      * Stage 3: (NOT IMPLEMENTED HERE) 2 additional RX sites will be added.


    **Beam options:**

      * gauss: Gaussian tapered beam model :func:`antenna_library.planar_beam`.
      * interp: Interpolated array pattern.
      * array: Ideal summation of all antennas in the array :func:`antenna_library.e3d_array_beam_stage1` and :func:`antenna_library.e3d_array_beam`.

    '''
    if beam=='array':
        tx_beam_ski = alib.e3d_array_stage1.copy()
        rx_beam_ski = alib.e3d_array_stage2.copy()
        rx_beam_kar = alib.e3d_array_stage2.copy()
        rx_beam_kai = alib.e3d_array_stage2.copy()
    elif beam=='interp':
        tx_intp, rx_intp = eiscat3d_interp()
        tx_beam_ski, = tx_intp
        rx_beam_ski, rx_beam_kar, rx_beam_kai = rx_intp

    ski_lat = 69.34023844
    ski_lon = 20.313166
    ski_alt = 0.0
    ski = RX(
        lat = ski_lat,
        lon = ski_lon,
        alt = ski_alt,
        min_elevation = 30.0,
        noise = 150,
        beam = rx_beam_ski,
    )
    dwell_time = 0.1
    ski_tx = TX(
        lat = ski_lat,
        lon = ski_lon,
        alt = ski_alt,
        min_elevation = 30.0,
        beam = tx_beam_ski,
        power = 5e6, # 5 MW
        bandwidth = 100e3, # 100 kHz tx bandwidth
        duty_cycle = 0.25, # 25% duty-cycle
        pulse_length=1920e-6,
        ipp=10e-3,
        n_ipp=int(dwell_time/10e-3),
    )

    kar_lat = 68.463862
    kar_lon = 22.458859
    kar_alt = 0.0
    kar = RX(
        lat = kar_lat,
        lon = kar_lon,
        alt = kar_alt,
        min_elevation = 30.0,
        noise = 150,
        beam = rx_beam_kar,
    )

    kai_lat = 68.148205
    kai_lon = 19.769894
    kai_alt = 0.0
    kai = RX(
        lat = kai_lat,
        lon = kai_lon,
        alt = kai_alt,
        min_elevation = 30.0,
        noise = 150,
        beam = rx_beam_kai,
    )
    # define transmit and receive antennas for a radar network.
    tx=[ski_tx]
    rx=[ski, kar, kai]

    eiscat3d = Radar(
        tx=tx, 
        rx=rx, 
        max_off_axis=120.0, 
        min_SNRdb=10.0,
    )
    return eiscat3d

