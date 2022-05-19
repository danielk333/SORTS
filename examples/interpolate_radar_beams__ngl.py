'''
Interpolated Antenna array gain
================================
'''

import numpy as np

import pyant
from sorts.radar.system import instances as rlib

radar = rlib.eiscat3d_demonstrator

res = 500

tx_intp = []
for txi,tx in enumerate(radar.tx):
    tx_intp += [pyant.PlaneArrayInterp(
        azimuth=tx.beam.azimuth,
        elevation=tx.beam.elevation, 
        frequency=tx.beam.frequency,
    )]
    tx_intp[-1].generate_interpolation(tx.beam, resolution=res)


rx_intp = []
for rxi,rx in enumerate(radar.rx):
    rx_intp += [pyant.PlaneArrayInterp(
        azimuth=rx.beam.azimuth,
        elevation=rx.beam.elevation, 
        frequency=rx.beam.frequency,
    )]
    rx_intp[-1].generate_interpolation(rx.beam, resolution=res)

print (rx_intp)