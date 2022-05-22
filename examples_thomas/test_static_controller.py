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

for tx in eiscat3d.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in eiscat3d.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')
    
for txi, tx in enumerate(eiscat3d.tx):
    points_tx = controls["beam_direction_tx"][txi]

    for rxi, rx in enumerate(eiscat3d.rx):
        points_rx = controls["beam_direction_rx"][rxi][txi]
        
        for ir in range(len(points_rx)):
            for j in range(len(points_rx[ir, 0])):
                ktx = points_tx[:, j]
                krx = points_rx[ir, :, j]
            
                a=np.dot(ktx, krx)
                
                if abs(abs(a)-1) < 0.0001:
                    ecef = points_rx[ir, :, j]
                else:
                    ecef = tx.ecef + np.dot(tx.ecef - rx.ecef, a*krx - ktx)/(1 - a**2)*ktx
                
                    ax.plot([tx.ecef[0], ecef[0]], [tx.ecef[1], ecef[1]], [tx.ecef[2], ecef[2]], 'r-', alpha=0.15)
                    ax.plot([rx.ecef[0], ecef[0]], [rx.ecef[1], ecef[1]], [rx.ecef[2], ecef[2]], 'g-', alpha=0.15)

dr = 600e3
ax.set_xlim([eiscat3d.tx[0].ecef[0]-dr, eiscat3d.tx[0].ecef[0]+dr])
ax.set_ylim([eiscat3d.tx[0].ecef[1]-dr, eiscat3d.tx[0].ecef[1]+dr])
ax.set_zlim([eiscat3d.tx[0].ecef[2]-dr, eiscat3d.tx[0].ecef[2]+dr])

plt.show()

print(p)
