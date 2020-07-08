#!/usr/bin/env python

'''

'''

import matplotlib.pyplot as plt

import sorts
import pyant

from sorts.radar import eiscat3d

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