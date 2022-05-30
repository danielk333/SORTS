#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:15:39 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

from sorts.radar.scans import Fence
from sorts.radar.system import instances
from sorts.radar import controllers

from sorts.common import profiling
from sorts import plotting

# Logger
logger = profiling.get_logger('scanning')
logger.info("test_scan_controller -> starting test/example script execution\n")
logger.info("test_scan_controller -> Setting up variables :")

# Computation / test setup
end_t = 24*3600
nbplots = 1
t_slice = 0.1
max_points = 100
log_array_sizes = True

logger.info(f"test_scan_controller:computation_variables -> end_t = {end_t}")
logger.info(f"test_scan_controller:computation_variables -> nbplots = {nbplots}")
logger.info(f"test_scan_controller:computation_variables -> t_slice = {t_slice}")
logger.info(f"test_scan_controller:computation_variables -> max_points = {max_points}")
logger.info(f"test_scan_controller:computation_variables -> log_array_sizes = {log_array_sizes}\n")

# Scan type definition
scan = Fence(azimuth=90, min_elevation=30, dwell=0.1, num=50)
logger.info(f"test_scan_controller -> scan initialized (Fence scan) : {scan}")

# RADAR definition
eiscat3d = instances.eiscat3d
logger.info(f"test_scan_controller -> radar initialized (eiscat3D) : {eiscat3d}")

# Profiler
p = profiling.Profiler()
logger.info(f"test_scan_controller -> profiler initialized : {p.__class__}")

# generate the beam orientation controls
p.start("test_scan_controller:Total")
p.start("test_scan_controller:compute_controls")

# instanciate the scanning controller 
scanner_ctrl = controllers.Scanner(profiler=p, logger=logger)
logger.info(f"test_scan_controller -> controller initialized : {scanner_ctrl}")

t = np.arange(0, end_t, scan.dwell())
logger.info(f"test_scan_controller -> generating time points - size={len(t)}")

logger.info("test_scan_controller -> generating controls")
controls = scanner_ctrl.generate_controls(t, eiscat3d, scan, max_points=max_points)
logger.info("test_scan_controller -> controls generated ! ")
logger.info(f"test_scan_controller -> size = {np.shape(controls['t'])[0]}")

p.stop("test_scan_controller:compute_controls")

# creating plots
if nbplots > 0:
    plt_ids = np.linspace(0, int(end_t/t_slice/max_points)-1, nbplots, dtype=int)

# compute control values
p.start("test_scan_controller:retreiving_control_values")

logger.info("test_static_controller -> retreiving controls : ")
for i in range(len(controls["beam_orientation"])):
    ctrl = next(controls["beam_orientation"][i])
    
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

    del ctrl

p.stop("test_scan_controller:retreiving_control_values")
p.stop("test_scan_controller:Total")

logger.info("test_scan_controller -> execution finised !")

plt.show()
print(p)
