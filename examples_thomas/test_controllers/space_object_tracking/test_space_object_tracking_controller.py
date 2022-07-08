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
from sorts import SpaceObjectTracker
from sorts.radar import controllers

from sorts.common import profiling
from sorts import plotting
from sorts import space_object

from sorts.targets.propagator import Kepler
from sorts import find_simultaneous_passes, equidistant_sampling

# Profiler
p = profiling.Profiler()
logger = profiling.get_logger('test_space_object_tracking_controller')

logger.info("test_space_object_tracking_controller: starting script")

p.start("test_space_object_tracking_controller:Total")    
p.start("test_space_object_tracking_controller:Initialization")  
# RADAR definition
eiscat3d = instances.eiscat3d
logger.info(f"test_space_object_tracking_controller -> intialized radar instance : {eiscat3d}")

# scheduler and controller definition
tracking_period = 50
t_slice = 2
t_start = 0
t_end = 3600*24

epoch = 53005.0

logger.info("test_space_object_tracking_controller -> controller/scheduler parameters :")
logger.info(f"test_space_object_tracking_controller -> tracking_period={tracking_period}")
logger.info(f"test_space_object_tracking_controller -> t_slice={t_slice}")
logger.info(f"test_space_object_tracking_controller -> t_start={t_start}")
logger.info(f"test_space_object_tracking_controller -> t_end={t_end}")
logger.info(f"test_space_object_tracking_controller -> epoch={epoch}")

t_tracking = np.arange(t_start, t_end, tracking_period)
logger.info(f"test_space_object_tracking_controller -> tracking time points created : {t_tracking}")

so_tracking_controller = SpaceObjectTracker(logger=logger, profiler=p)
logger.info(f"test_space_object_tracking_controller -> intialized tracking scheduler instance : {so_tracking_controller}")

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
    logger.info(f"test_space_object_tracking_controller -> space object {so_id} create")

logger.info("test_space_object_tracking_controller -> Initialization done")
p.stop("test_space_object_tracking_controller:Initialization")  

p.start("test_space_object_tracking_controller:Computations")  
logger.info("test_space_object_tracking_controller -> Generating tracking controls")
controls = so_tracking_controller.generate_controls(t_tracking, eiscat3d, space_objects, epoch, t_slice, priority=priority, save_states=True)
logger.info(f"test_space_object_tracking_controller -> Tracking controls generated")
p.stop("test_space_object_tracking_controller:Computations")  

# plotting results
logger.info("test_space_object_tracking_controller -> plotting results")
p.start("test_space_object_tracking_controller:Plotting")  

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting station ECEF positions
logger.info("test_space_object_tracking_controller -> plotting earth grid")
plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

# Plotting station ECEF positions
logger.info("test_space_object_tracking_controller -> plotting radar stations")
for tx in eiscat3d.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in eiscat3d.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')       

logger.info("test_space_object_tracking_controller -> plotting tracked object states")

ecef_tracking = controls.space_objects_states
object_ids = controls.state_priorities

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

logger.info("test_space_object_tracking_controller -> plotting radar pointing directions")
for period_id in range(controls.n_periods):
    ax = plotting.plot_beam_directions(controls.get_pdirs(period_id), eiscat3d, ax=ax, zoom_level=0.6, azimuth=10, elevation=20)

p.stop("test_space_object_tracking_controller:Plotting")  
p.stop("test_space_object_tracking_controller:Total")    
logger.info("test_space_object_tracking_controller -> execution finised !")

print(p)

plt.show()