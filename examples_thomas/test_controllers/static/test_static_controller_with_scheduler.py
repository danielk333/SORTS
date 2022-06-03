#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 26 09:38:07 2022

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

# setting up logger 
logger = profiling.get_logger('test_scan_controller_w_scheduler')
logger.info("test_scan_controller_w_scheduler -> starting test/example script execution\n")
logger.info("test_scan_controller_w_scheduler -> Setting up variables :")

# Computation / test setup
end_t = 24*3600
nbplots = 1
t_slice = 0.1
log_array_sizes = True

logger.info(f"test_static_controller_w_scheduler:computation_variables -> end_t = {end_t}")
logger.info(f"test_static_controller_w_scheduler:computation_variables -> nbplots = {nbplots}")
logger.info(f"test_static_controller_w_scheduler:computation_variables -> t_slice = {t_slice}")
logger.info(f"test_static_controller_w_scheduler:computation_variables -> log_array_sizes = {log_array_sizes}\n")

# RADAR definition
eiscat3d = instances.eiscat3d

logger.info("test_static_controller_w_scheduler -> radar initialized : eiscat3D\n")

# scheduler properties
t0 = 0
scheduler_period = 120 # [s] -> 2 minutes - higher slices reduce computation time, but beware the RAM usage

logger.info("test_static_controller_w_scheduler -> scheduler initialized:")
logger.info(f"test_static_controller_w_scheduler:scheduler_variables -> t0 = {t0}")
logger.info(f"test_static_controller_w_scheduler:scheduler_variables -> scheduler_period = {scheduler_period}\n")

# Profiler
p = profiling.Profiler()
logger.info("test_static_controller_w_scheduler -> profiler initialized")

# starting profiler
p.start("Total")
p.start("test_static_controller_w_scheduler:computing_controls")

# create scheduler and controller
scheduler = scheduler.Scheduler(eiscat3d, t0, scheduler_period=scheduler_period)
logger.info(f"test_static_controller_w_scheduler -> scheduler initialized : {scheduler}")

static_controller = controllers.Static(profiler=p, logger=logger)
logger.info(f"test_static_controller_w_scheduler -> controller initialized : {static_controller}")

logger.info("test_static_controller_w_scheduler -> generating controls")
t_slice = 0.1
t = np.arange(0, end_t, t_slice)
logger.info(f"test_static_controller_w_scheduler -> generating time points - size={len(t)}")

controls = static_controller.generate_controls(t, eiscat3d, t_slice=t_slice, scheduler=scheduler)
logger.info("test_static_controller_w_scheduler -> controls generated ! ")
logger.info(f"test_static_controller_w_scheduler -> size = {np.shape(controls['t'])[0]}")

# plot the generated controls
p.stop("test_static_controller_w_scheduler:computing_controls")
p.start("test_static_controller_w_scheduler:retreiving_control_values")

plt_ids = np.linspace(0, int(end_t/t_slice), nbplots, dtype=int)

if nbplots > 0:
    plt_ids = np.linspace(0, int(end_t/t_slice/len(controls["t"][0]))-1, nbplots, dtype=int)
else:
    plt_ids = None

logger.info("test_static_controller_w_scheduler -> retreiving controls : ")
for ctrl_id in range(len(controls["t"])):
    ctrl = next(controls["pointing_direction"])
    
    if log_array_sizes is True:
        logger.info(f"test_static_controller_w_scheduler: controls {ctrl_id} - size : {(ctrl['tx'].itemsize*np.size(ctrl['tx']) + ctrl['rx'].itemsize*np.size(ctrl['rx']))/1e6} Mb")
    
    if plt_ids is not None:
        if ctrl_id in plt_ids:
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
    
p.stop("test_static_controller_w_scheduler:retreiving_control_values")
p.stop("Total")

logger.info("test_static_controller_w_scheduler -> execution finised !")

print(p)
plt.show()

