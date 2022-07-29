#!/usr/bin/env python

'''
===================================
Predefined EISCAT 3D Demonstrator
===================================

This example imports and plots the gain pattern of the predifined eiscat3d demonstrator.
'''

import matplotlib.pyplot as plt

import sorts
import pyant

import sorts

# initialize radar instances 
radar = sorts.radars.eiscat3d_demonstrator
radar_interp = sorts.radars.eiscat3d_demonstrator_interp # interpolated gain pattern for faster computations

fig = plt.figure(figsize=(15,6))
axes = [
    fig.add_subplot(131, projection='3d'),
    fig.add_subplot(132), 
    fig.add_subplot(133),
]

# plot antenna array configuration of the tx station
pyant.plotting.antenna_configuration(radar.tx[0].beam.antennas, ax=axes[0])

# plot gain pattern (full computation) of the tx station
pyant.plotting.gain_heatmap(radar.tx[0].beam, resolution=100, min_elevation=30.0, ax=axes[1])
axes[1].set_title('Array gain')

# plot gain pattern (interpolation) of the tx station
pyant.plotting.gain_heatmap(radar_interp.tx[0].beam, resolution=100, min_elevation=30.0, ax=axes[2])
axes[2].set_title('Interpolated gain')

pyant.plotting.show()