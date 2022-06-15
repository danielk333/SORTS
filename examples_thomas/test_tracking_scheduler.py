#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 08:16:43 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 100000000

from sorts.radar.system import instances
from sorts import TrackingScheduler
from sorts.radar import controllers

from sorts.common import profiling
from sorts import plotting
from sorts import space_object

from sorts.targets.propagator import Kepler
from sorts import find_simultaneous_passes, equidistant_sampling

# Profiler
p = profiling.Profiler()
logger = profiling.get_logger('tracking_scheduler')

logger.info("starting script")

p.start("test_tracking_scheduler:Total")    
p.start("test_tracking_scheduler:Initialization")  
# RADAR definition
eiscat3d = instances.eiscat3d
logger.info(f"test_tracking_scheduler -> intialized radar instance : {eiscat3d}")

# scheduler and controller definition
tracking_period = 50
t_slice = 2
t_start = 0
t_end = 100000

epoch = 53005.0

logger.info("test_tracking_scheduler -> controller/scheduler parameters :")
logger.info(f"test_tracking_scheduler -> tracking_period={tracking_period}")
logger.info(f"test_tracking_scheduler -> t_slice={t_slice}")
logger.info(f"test_tracking_scheduler -> t_start={t_start}")
logger.info(f"test_tracking_scheduler -> t_end={t_end}")
logger.info(f"test_tracking_scheduler -> epoch={epoch}")

t_tracking = np.arange(t_start, t_end, tracking_period)
logger.info(f"test_tracking_scheduler -> tracking time points created : {t_tracking}")

tracker_controller = controllers.Tracker(logger=logger, profiler=p)
logger.info(f"test_tracking_scheduler -> intialized tracking controller instance : {tracker_controller}")

tracking_scheduler = TrackingScheduler(logger=logger, profiler=p)
logger.info(f"test_tracking_scheduler -> intialized tracking scheduler instance : {tracking_scheduler}")

# Propagator
Prop_cls = Kepler
Prop_opts = dict(
    settings = dict(
        out_frame='ITRS',
        in_frame='TEME',
    ),
)

# Object definition
# Creating space object
# Object properties
orbits_a = np.array([7200, 7200, 8500, 12000, 10000])*1e3 # m
orbits_i = np.array([80, 80, 105, 105, 80]) # deg
orbits_raan = np.array([86, 86, 160, 180, 90]) # deg
orbits_aop = np.array([0, 0, 50, 40, 55]) # deg
orbits_mu0 = np.array([60, 50, 5, 30, 8]) # deg

priority = np.array([3, 2, 0, 1, 4])

space_objects = []
for so_id in range(len(orbits_a)):
    space_objects.append(space_object.SpaceObject(
            Prop_cls,
            propagator_options = Prop_opts,
            a = orbits_a[so_id], 
            e = 0.1,
            i = orbits_i[so_id],
            raan = orbits_raan[so_id],
            aop = orbits_aop[so_id],
            mu0 = orbits_mu0[so_id],
            
            epoch = epoch,
            parameters = dict(
                d = 0.1,
            ),
        ))
    logger.info(f"test_tracking_scheduler -> space object {so_id} create")

logger.info("test_tracking_scheduler -> Initialization done")
p.stop("test_tracking_scheduler:Initialization")  

p.start("test_tracking_scheduler:Computations")  
logger.info("test_tracking_scheduler -> Generating tracking schedule")
t_tracking, ecef_tracking, object_ids = tracking_scheduler.generate_schedule(t_tracking, space_objects, eiscat3d, epoch, priority=priority)
logger.info(f"test_tracking_scheduler -> Tracking schedule generated")

logger.info("test_tracking_scheduler -> Generating tracking controls")
controls = tracker_controller.generate_controls(t_tracking.copy(), eiscat3d, t_tracking, ecef_tracking, t_slice=t_slice, max_points=1000, priority=0, states_per_slice=10)
logger.info(f"test_tracking_scheduler -> Tracking controls generated")
p.stop("test_tracking_scheduler:Computations")  

# plotting results
logger.info("test_tracking_scheduler -> plotting results")
p.start("test_tracking_scheduler:Plotting")  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting station ECEF positions
logger.info("test_tracking_scheduler -> plotting earth grid")
plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

# Plotting station ECEF positions
logger.info("test_tracking_scheduler -> plotting radar stations")
for tx in eiscat3d.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in eiscat3d.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')       

logger.info("test_tracking_scheduler -> plotting tracked object states")
transition_ids = np.where(np.abs(object_ids[1:] - object_ids[:-1]) > 0)[0]+1
states_split = []
for i in range(len(transition_ids)+1):
    if i == 0:
        i_start = 0
    else:
        i_start = transition_ids[i-1]

    if i == len(transition_ids):
        i_end = len(t_tracking)-1
    else:
        i_end = transition_ids[i]

    states_split.append(ecef_tracking[:, i_start:i_end])

    ax.plot(ecef_tracking[:, i_start:i_end][0], ecef_tracking[:, i_start:i_end][1], ecef_tracking[:, i_start:i_end][2], '-b')

logger.info("test_tracking_scheduler -> plotting radar pointing directions")
for period_id in range(len(controls["t"])):
    ax = plotting.plot_beam_directions(next(controls["pointing_direction"]), eiscat3d, ax=ax, zoom_level=0.6, azimuth=10, elevation=20)

p.stop("test_tracking_scheduler:Plotting")  
p.stop("test_tracking_scheduler:Total")    
logger.info("test_tracking_scheduler -> execution finised !")

print(p)

plt.show()