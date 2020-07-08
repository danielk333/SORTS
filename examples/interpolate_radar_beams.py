'''
Interpolated Antenna array gain
================================
'''

import numpy as np

import pyant
import sorts.radar.instances as rlib

res = 500

tx_intp = []
for txi,tx in enumerate(rlib.eiscat3d.tx):
    tx_intp += [pyant.PlaneArrayInterp(
        azimuth=tx.beam.azimuth,
        elevation=tx.beam.elevation, 
        frequency=tx.beam.frequency,
    )]
    tx_intp[-1].generate_interpolation(tx.beam, resolution=res)
    tx_intp[-1].save(f'./sorts/data/e3d_tx{txi}_res{res}_interp')
    print('saved')

rx_intp = []
for rxi,rx in enumerate(rlib.eiscat3d.rx):
    rx_intp += [pyant.PlaneArrayInterp(
        azimuth=rx.beam.azimuth,
        elevation=rx.beam.elevation, 
        frequency=rx.beam.frequency,
    )]
    rx_intp[-1].generate_interpolation(rx.beam, resolution=res)
    rx_intp[-1].save(f'./sorts/data/e3d_rx{rxi}_res{res}_interp')
    print('saved')

