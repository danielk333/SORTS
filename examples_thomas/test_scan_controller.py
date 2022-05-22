#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:15:39 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

from sorts.radar.scans import Fence
from sorts.radar.system import instances
from sorts.radar import controllers
from sorts.common import profiling
from sorts import plotting

# Computation / test setup
end_t = 24*3600
nbplots = 0
t_slice = 0.1
max_points = 100

# Scan type definition
scan = Fence(azimuth=90, min_elevation=30, dwell=t_slice, num=50)

# RADAR definition
eiscat3d = instances.eiscat3d
    
# Profiler
p = profiling.Profiler()
logger = profiling.get_logger('scanning')

# generate the beam orientation controls
p.start("test_scan_controller:Total")
p.start("test_scan_controller:compute_controls")

# instanciate the scanning controller 
scanner_ctrl = controllers.Scanner(profiler=p, logger=logger)
t = np.arange(0, end_t, scan.dwell())

controls = scanner_ctrl.generate_controls(t, eiscat3d, scan, max_points=max_points)

p.stop("test_scan_controller:compute_controls")

plt_ids = np.linspace(0, int(end_t/t_slice/max_points)-1, nbplots, dtype=int)

# compute control values
p.start("test_scan_controller:compute_sub_controls")

for i in range(len(controls["beam_orientation"])):
    ctrl = next(controls["beam_orientation"][i])
    #print(f"control sub-array {i} size : {(ctrl['tx'].itemsize*np.size(ctrl['tx']) + ctrl['rx'].itemsize*np.size(ctrl['rx']))/1e6} Mb")
    
    if i in plt_ids:
        plotting.plot_beam_directions(ctrl, eiscat3d, logger=logger, profiler=p, zoom_level=0.95)

    del ctrl

print(f"plot indices : {plt_ids}")

p.stop("test_scan_controller:compute_sub_controls")
p.stop("test_scan_controller:Total")

plt.show()

print(p)