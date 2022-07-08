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

from sorts.common import profiling
from sorts import plotting
from sorts import space_object

from sorts import StaticPriorityScheduler
from sorts.common import interpolation
from sorts.targets.propagator import Kepler
from sorts import find_simultaneous_passes, equidistant_sampling

plt.rcParams['agg.path.chunksize'] = 100000000

# Profiler
p = profiling.Profiler()

p.start("test_static_priority_scheduler:Total")    
p.start("test_static_priority_scheduler:Init")

# Logger
logger = profiling.get_logger('test_static_priority_scheduler')
logger.info("test_static_priority_scheduler -> starting test/example script execution\n")
logger.info("test_static_priority_scheduler -> Setting up variables :")

# Computation / test setup
end_t = 100
nbplots = 1
log_array_sizes = True

# scheduler properties
t0 = 0
scheduler_period = 50 # [s] -> 2 minutes - can go up to 10mins or more depending on the available RAM

logger.info("test_static_priority_scheduler -> scheduler initialized:")
logger.info(f"test_static_priority_scheduler:scheduler_variables -> t0 = {t0}")
logger.info(f"test_static_priority_scheduler:scheduler_variables -> scheduler_period = {scheduler_period}\n")

# RADAR definition
eiscat3d = instances.eiscat3d

logger.info("test_static_priority_scheduler -> radar initialized : eiscat3D\n")

# generate the beam orientation controls
# create scheduler
scheduler = StaticPriorityScheduler(eiscat3d, t0, scheduler_period, profiler=p, logger=logger)

logger.info("test_static_priority_scheduler -> profiler initialized")

# instanciate the scanning controller 
scanner_ctrl = controllers.Scanner(profiler=p, logger=logger)
logger.info("test_static_priority_scheduler -> controller & scheduler created")

logger.info("test_static_priority_scheduler -> generating controls")

# compute time array
p.stop("test_static_priority_scheduler:Init")
p.start("test_static_priority_scheduler:compute_controls")

# compute controls

# SCANNER
# controls parameters 
controls_period = np.array([1, 5, 20])
controls_t_slice = np.array([0.5, 2.5, 5])

controls_priority = np.array([2, 1, 0], dtype=int)+1
controls_start = np.array([0, 4, 49.5], dtype=float)
controls_end = np.array([end_t, end_t, end_t], dtype=float)

controls_az = np.array([90, 45, 80])
controls_el = np.array([10, 20, 30])

scans = []
for i in range(len(controls_period)):
    scans.append(Fence(azimuth=controls_az[i], min_elevation=controls_el[i], dwell=controls_t_slice[i], num=int(controls_end[i]/controls_t_slice[i])))

logger.info(f"test_static_priority_scheduler:computation_variables -> end_t = {end_t}")
logger.info(f"test_static_priority_scheduler:computation_variables -> nbplots = {nbplots}")
logger.info(f"test_static_priority_scheduler:computation_variables -> controls_t_slice = {controls_t_slice}")
logger.info(f"test_static_priority_scheduler:computation_variables -> controls_period = {controls_period}")
logger.info(f"test_static_priority_scheduler:computation_variables -> controls_priority = {controls_priority}")
logger.info(f"test_static_priority_scheduler:computation_variables -> controls_az = {controls_az}")
logger.info(f"test_static_priority_scheduler:computation_variables -> controls_el = {controls_el}")
logger.info(f"test_static_priority_scheduler:computation_variables -> log_array_sizes = {log_array_sizes}\n")
logger.info(f"test_static_priority_scheduler:computation_variables -> controls_start = {controls_start}\n")
logger.info(f"test_static_priority_scheduler:computation_variables -> controls_start = {controls_end}\n")

controls = []

for i in range(len(controls_period)):
    t = np.arange(controls_start[i], controls_end[i], controls_period[i])
    logger.info(f"test_static_priority_scheduler -> generating time points - size={len(t)}")
    controls.append(scanner_ctrl.generate_controls(t, eiscat3d, scans[i], scheduler=scheduler, priority=controls_priority[i]))

# TRACKER
# Object definition
# Propagator
Prop_cls = Kepler
Prop_opts = dict(
    settings = dict(
        out_frame='ITRS',
        in_frame='TEME',
    ),
)

tracking_period = 15
t_slice = 2
obj_id = 0

p.start('test_static_priority_scheduler:object_initialization')

# Creating space object
# Object properties
orbits_a = np.array([7200, 8500, 12000, 10000])*1e3 # km
orbits_i = np.array([80, 105, 105, 80]) # deg
orbits_raan = np.array([86, 160, 180, 90]) # deg
orbits_aop = np.array([0, 50, 40, 55]) # deg
orbits_mu0 = np.array([60, 5, 30, 8]) # deg

space_object = space_object.SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = orbits_a[obj_id], 
        e = 0.1,
        i = orbits_i[obj_id],
        raan = orbits_raan[obj_id],
        aop = orbits_aop[obj_id],
        mu0 = orbits_mu0[obj_id],
        
        epoch = 53005.0,
        parameters = dict(
            d = 0.1,
        ),
    )

p.stop('test_static_priority_scheduler:object_initialization')
logger.info("test_static_priority_scheduler -> object created :")
logger.info(f"test_static_priority_scheduler -> {space_object}")

logger.info("test_static_priority_scheduler -> sampling equidistant states on the orbit")


# create state time array
p.start('test_static_priority_scheduler:equidistant_sampling')
t_states = equidistant_sampling(
    orbit = space_object.state, 
    start_t = 0, 
    end_t = end_t, 
    max_dpos=50e3,
)
p.stop('test_static_priority_scheduler:equidistant_sampling')

# get object states in ECEF frame
p.start('test_static_priority_scheduler:get_state')
object_states = space_object.get_state(t_states)
p.stop('test_static_priority_scheduler:get_state')

logger.info("test_static_priority_scheduler -> object states computation done ! ")
logger.info(f"test_static_priority_scheduler -> t_states -> {t_states.shape}")

fig = plt.figure(dpi=300, figsize=(5, 5))
ax = fig.add_subplot(111, projection='3d')

# Plotting station ECEF positions
plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

# Plotting station ECEF positions
for tx in eiscat3d.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in eiscat3d.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

# reduce state array
p.start('test_static_priority_scheduler:find_simultaneous_passes')
eiscat_passes = find_simultaneous_passes(t_states, object_states, [*eiscat3d.tx, *eiscat3d.rx])
p.stop('test_static_priority_scheduler:find_simultaneous_passes')
logger.info(f"test_static_priority_scheduler -> Passes : eiscat_passes={eiscat_passes}")

p.start('test_static_priority_scheduler:generate_tracking_controls')
tracker_controller = controllers.Tracker(logger=logger, profiler=p)

for pass_id in range(np.shape(eiscat_passes)[0]):
    logger.info(f"test_static_priority_scheduler -> Computing tracking controls for pass {pass_id}:")

    tracking_states = object_states[:, eiscat_passes[pass_id].inds]
    t_states_i = t_states[eiscat_passes[pass_id].inds]

    ax.plot(tracking_states[0], tracking_states[1], tracking_states[2], "-b")
    
    p.start('test_static_priority_scheduler:intitialize_controller')
    t_controller = np.arange(t_states_i[0], t_states_i[-1]+tracking_period, tracking_period)
    p.stop('test_static_priority_scheduler:intitialize_controller')

    controls.append(tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, scheduler=scheduler, priority=0, states_per_slice=20, interpolator=interpolation.Linear))
    
    logger.info("test_static_priority_scheduler -> Controls generated")
p.stop('test_static_priority_scheduler:generate_tracking_controls')

logger.info("test_static_priority_scheduler -> generating final_control_sequence")
final_control_sequence = scheduler.run(controls, t_start=29, t_end=70)
logger.info("test_tracker_controller -> final_control_sequence generated")


logger.info("test_static_priority_scheduler -> plotting final_control_sequence uptime")
figs = plotting.plot_scheduler_control_sequence(controls, final_control_sequence, scheduler, logger=logger, profiler=p)

fmts = ["b-", "m-", "k-", "-c"]

logger.info("test_static_priority_scheduler -> plotting final_control_sequence directions")
for period_id in range(final_control_sequence.n_periods):
    ax = plotting.plot_beam_directions(final_control_sequence.get_pdirs(period_id), eiscat3d, ax=ax)

    for ctrl_i in range(len(controls)):
        ax = plotting.plot_beam_directions(controls[ctrl_i].get_pdirs(period_id), eiscat3d, ax=ax, fmt=fmts[ctrl_i], linewidth_rx=0.08, linewidth_tx=0.08, alpha=0.8)

p.stop("test_static_priority_scheduler:compute_controls")

logger.info("test_static_priority_scheduler -> execution finised !")

print(p)

plt.show()