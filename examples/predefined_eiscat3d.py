#!/usr/bin/env python

"""
Predefined EISCAT 3D Radar
================================
"""
from pprint import pprint
import matplotlib.pyplot as plt

import sorts
import pyant.plotting as aplt

print("Radars:")
pprint(sorts.list_radars())

eiscat3d = sorts.get_radar("eiscat3d", "stage1-array")
eiscat3d_interp = sorts.get_radar(
    "eiscat3d",
    "stage1-interp",
    resolution=(200, 200, None),
    min_elevation=80.0,
)

fig, axes = plt.subplots(1, 2, figsize=(15, 15))

aplt.gain_heatmap(eiscat3d.tx[0].beam, ax=axes[0], resolution=100, min_elevation=80.0)
axes[0].set_title("EISCAT 3D Stage 1 array model")

aplt.gain_heatmap(eiscat3d_interp.tx[0].beam, ax=axes[1], resolution=100, min_elevation=80.0)
axes[1].set_title("EISCAT 3D Stage 1 interpolated array model")


fig = plt.figure(figsize=(15, 15))
axes = [
    [
        fig.add_subplot(221, projection="3d"),
        fig.add_subplot(222, projection="3d"),
    ],
    [
        fig.add_subplot(223),
        fig.add_subplot(224),
    ],
]

aplt.antenna_configuration(eiscat3d.tx[0].beam.antennas, ax=axes[0][0])
aplt.gain_heatmap(eiscat3d_interp.tx[0].beam, resolution=100, min_elevation=80.0, ax=axes[1][0])

aplt.antenna_configuration(eiscat3d.rx[0].beam.antennas, ax=axes[0][1])
aplt.gain_heatmap(eiscat3d_interp.rx[0].beam, resolution=100, min_elevation=80.0, ax=axes[1][1])

plt.show()
