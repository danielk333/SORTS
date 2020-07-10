#!/usr/bin/env python

'''
Predefined EISCAT 3D Radar
================================
'''

import matplotlib.pyplot as plt

import sorts
import pyant

import sorts
eiscat3d = sorts.radars.eiscat3d

#dig trough the documentation to find how to generate alternative configurations of the predefined instances
eiscat3d_interp = sorts.radars.eiscat3d_interp

pyant.plotting.gain_heatmap(eiscat3d_interp.tx[0].beam, resolution=100, min_elevation=80.0)

fig = plt.figure(figsize=(15,15))
axes = [
    [
        fig.add_subplot(221, projection='3d'), 
        fig.add_subplot(222, projection='3d'),
    ],
    [
        fig.add_subplot(223), 
        fig.add_subplot(224),
    ]
]

pyant.plotting.antenna_configuration(eiscat3d.tx[0].beam.antennas, ax=axes[0][0])
pyant.plotting.gain_heatmap(eiscat3d.tx[0].beam, resolution=100, min_elevation=80.0, ax=axes[1][0])

pyant.plotting.antenna_configuration(eiscat3d.rx[0].beam.antennas, ax=axes[0][1])
pyant.plotting.gain_heatmap(eiscat3d.rx[0].beam, resolution=100, min_elevation=80.0, ax=axes[1][1])

pyant.plotting.show()