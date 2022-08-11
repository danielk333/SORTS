#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 08:16:43 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

from sorts.radar.scans import Fence
from sorts.radar.system import instances
from sorts.radar import controllers
from sorts.radar import scheduler

from sorts.common import profiling
from sorts import plotting

plt.rcParams['agg.path.chunksize'] = 100000000

# Profiler
p = profiling.Profiler()

p.start("test_scan_controller_w_scheduler:Total")    
p.start("test_scan_controller_w_scheduler:Init")

# Logger
logger = profiling.get_logger('test_scan_controller_w_scheduler')
logger.info("test_scan_controller_w_scheduler -> starting test/example script execution\n")
logger.info("test_scan_controller_w_scheduler -> Setting up variables :")

# Computation / test setup
end_t = 24*3600
nbplots = 2
t_slice = 0.1
log_array_sizes = True

logger.info(f"test_scan_controller_w_scheduler:computation_variables -> end_t = {end_t}")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> nbplots = {nbplots}")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> t_slice = {t_slice}")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> log_array_sizes = {log_array_sizes}\n")

# scheduler properties
t0 = 0
scheduler_period = 120 # [s] -> 1 minutes - can go up to 10mins or more depending on the available RAM
scan = Fence(azimuth=90, min_elevation=30, dwell=t_slice, num=24*36)

logger.info("test_scan_controller_w_scheduler -> scheduler initialized:")
logger.info(f"test_scan_controller_w_scheduler:scheduler_variables -> t0 = {t0}")
logger.info(f"test_scan_controller_w_scheduler:scheduler_variables -> scheduler_period = {scheduler_period}\n")

# RADAR definition
eiscat3d = instances.eiscat3d
logger.info("test_scan_controller_w_scheduler -> radar initialized : eiscat3D\n")

# generate the beam orientation controls
# create scheduler
scheduler = scheduler.StaticPriorityScheduler(eiscat3d, t0, scheduler_period=scheduler_period)
logger.info("test_scan_controller_w_scheduler -> profiler initialized")

# instanciate the scanning controller 
scanner_ctrl = controllers.Scanner(profiler=p, logger=logger)
logger.info("test_scan_controller_w_scheduler -> controller & scheduler created")

# compute time array
t = np.arange(0, end_t, scan.dwell())
logger.info(f"test_scan_controller_w_scheduler -> generating time points - size={len(t)}")
p.stop("test_scan_controller_w_scheduler:Init")

p.start("test_scan_controller_w_scheduler:compute_controls")

# compute controls
logger.info("test_scan_controller_w_scheduler -> generating controls")
controls = scanner_ctrl.generate_controls(t, eiscat3d, scan, scheduler=scheduler)
logger.info("test_scan_controller_w_scheduler -> controls generated ! ")
logger.info(f"test_scan_controller_w_scheduler -> size = {np.shape(controls.t)[0]}")

p.stop("test_scan_controller_w_scheduler:compute_controls")

if nbplots > 0:
    plt_ids = np.linspace(0, int(end_t/t_slice/len(controls.t[0]))-1, nbplots, dtype=int)
else:
    plt_ids = None

# compute control values
p.start("test_scan_controller_w_scheduler:retreiving_control_values")

logger.info("test_scan_controller_w_scheduler -> retreiving controls : ")
for period_id in range(controls.n_periods):
    ctrl = controls.get_pdirs(period_id)
    
    if log_array_sizes is True:
        logger.info(f"test_scan_controller_w_scheduler: controls {period_id} - size : {(ctrl['tx'].itemsize*np.size(ctrl['tx']) + ctrl['rx'].itemsize*np.size(ctrl['rx']))/1e6} Mb")
    
    if plt_ids is not None:
        if period_id in plt_ids:
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

p.stop("test_scan_controller_w_scheduler:retreiving_control_values")
p.stop("test_scan_controller_w_scheduler:Total")

logger.info("test_scan_controller_w_scheduler -> execution finised !")
print(p)

plt.show()