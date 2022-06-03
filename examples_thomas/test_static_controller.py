#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 22 14:54:50 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

from sorts.radar.system import instances
from sorts.radar import scheduler, controllers
from sorts import equidistant_sampling

from sorts.common import profiling
from sorts import plotting

import pyorb

# Computation / test setup
end_t = 24*3600
nbplots = 3

# RADAR definition
eiscat3d = instances.eiscat3d
    
# Profiler
p = profiling.Profiler()
logger = profiling.get_logger('scanning')

p.start("Total")
static_controller = controllers.Static(profiler=p, logger=logger)

print("Scanner init")
dwell = 0.1
t = np.arange(0, end_t, dwell)
controls = static_controller.generate_controls(t, eiscat3d, dwell=dwell)
print("Controls generated")

p.stop("Total")

fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
   
plotting.grid_earth(ax)

print("Plotting")
for ctrl_i in range(len(controls["t"])):
    ctrl = next(controls["pointing_direction"])
    
    if log_array_sizes is True:
        logger.info(f"test_scan_controller: controls {i} - size : {(ctrl['tx'].itemsize*np.size(ctrl['tx']) + ctrl['rx'].itemsize*np.size(ctrl['rx']))/1e6} Mb")

    if nbplots > 0:
        if i in plt_ids:
            fig = plt.figure(figsize=(15,15))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plotting station ECEF positions
            plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)
                
            
            # Plotting station ECEF positions
            for tx in eiscat3d.tx:
                ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
            for rx in eiscat3d.rx:
                ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

            ax = plotting.plot_beam_directions(ctrl, eiscat3d, ax=ax, logger=logger, profiler=p, zoom_level=0.95, azimuth=10, elevation=10)

plt.show()

print(p)