#!/usr/bin/env python

'''
=====================================
Using scans on alternative parameters
=====================================

This example
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import pyant
import sorts

from sorts.radar.scans import Fence
from sorts.radar import RadarController
from sorts import radar_controls
from sorts.targets import SpaceObject
from sorts.targets.propagator.pysgp4 import SGP4
from sorts import Profiler
from sorts.transformations import frames

Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)

radar = sorts.radars.tsdr_phased_fence

radar.tx[0].beam.phase_steering = 30.0

fig, axes = plt.subplots(2,2,figsize=(10,6),dpi=80)
axes = axes.flatten()
for i in range(4):
    pyant.plotting.gain_heatmap(
        radar.tx[0].beam, 
        resolution=901, 
        min_elevation=30.0, 
        ax=axes[i],
        ind = {
            "pointing":i,
        },
    )
    axes[i].set_title(f'Panel {i}: {int(radar.tx[0].beam.phase_steering)} deg steering')

radar.tx[0].beam.phase_steering = 0.0

for station in radar.tx + radar.rx:
    station.min_elevation = 30.0

scan = Beampark(azimuth=0.0, elevation=90.0, dwell=0.1)
end_t = 3600*24

p = Profiler()
logger = sorts.profiling.get_logger('scanning')

objs = [
    SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = 7200e3, 
        e = 0.02, 
        i = 75, 
        raan = 86,
        aop = 0,
        mu0 = 60,
        epoch = 53005.0,
        parameters = dict(
            d = 0.1,
        ),
    ),
]

controller = controllers.Scanner(t, radar, scan, r=[950e3], max_points=3600)



