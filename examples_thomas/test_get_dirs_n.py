#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 08:16:43 2022

@author: thomas
"""
import ctypes
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

from sorts import clibsorts

def get_dir_count_loop(dir_array_txrx):
    N = len(dir_array_txrx)
    dir_count = np.empty((N,), dtype=int)

    for ti in range(N):
        dir_count[i] = np.shape(dir_array_txrx[ti])[0]

    return dir_count

@np.vectorize
def get_dir_count_vec(dir_array_txrx):
    return np.shape(dir_array_txrx)[0]

def get_dir_count_c(dir_array_txrx):
    N = len(dir_array_txrx)
    dir_count = np.empty((N,), dtype=np.int32)

    
    def get_dir_count(ti):
        nonlocal dir_array_txrx
        return np.shape(dir_array_txrx[ti])[0]

    get_dir_count_vec_t = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_int)
    get_dir_count = get_dir_count_vec_t(get_dir_count)

    clibsorts.get_dir_count_c.argtypes = [
        get_dir_count_vec_t,
        np.ctypeslib.ndpointer(dtype=ctypes.c_int, ndim=dir_count.ndim, shape=dir_count.shape),
        ctypes.c_int,
    ]

    clibsorts.get_dir_count_c(get_dir_count, dir_count, ctypes.c_int(N))

    return dir_count

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
end_t = 100
nbplots = 1
log_array_sizes = True

# scheduler properties
t0 = 0
scheduler_period = 50 # [s] -> 2 minutes - can go up to 10mins or more depending on the available RAM

logger.info("test_scan_controller_w_scheduler -> scheduler initialized:")
logger.info(f"test_scan_controller_w_scheduler:scheduler_variables -> t0 = {t0}")
logger.info(f"test_scan_controller_w_scheduler:scheduler_variables -> scheduler_period = {scheduler_period}\n")

# RADAR definition
eiscat3d = instances.eiscat3d

logger.info("test_scan_controller_w_scheduler -> radar initialized : eiscat3D\n")

# generate the beam orientation controls
# create scheduler
scheduler = StaticPriorityScheduler(eiscat3d, t0, scheduler_period, profiler=p, logger=logger)

logger.info("test_scan_controller_w_scheduler -> profiler initialized")

# instanciate the scanning controller 
scanner_ctrl = controllers.Scanner(profiler=p, logger=logger)
logger.info("test_scan_controller_w_scheduler -> controller & scheduler created")

logger.info("test_scan_controller_w_scheduler -> generating controls")

# compute time array
p.stop("test_scan_controller_w_scheduler:Init")
p.start("test_scan_controller_w_scheduler:compute_controls")

# compute controls

# SCANNER
# controls parameters 
controls_period = np.array([1, 5, 20])*0.01
controls_t_slice = np.array([0.5, 2.5, 5])*0.01

controls_priority = np.array([2, 1, 0], dtype=int)+1
controls_start = np.array([0, 4, 49.5], dtype=float)
controls_end = np.array([end_t, end_t, end_t], dtype=float)

controls_az = np.array([90, 45, 80])
controls_el = np.array([10, 20, 30])

scans = []
for i in range(len(controls_period)):
    scans.append(Fence(azimuth=controls_az[i], min_elevation=controls_el[i], dwell=controls_t_slice[i], num=int(controls_end[i]/controls_t_slice[i])))

logger.info(f"test_scan_controller_w_scheduler:computation_variables -> end_t = {end_t}")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> nbplots = {nbplots}")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> controls_t_slice = {controls_t_slice}")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> controls_period = {controls_period}")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> controls_priority = {controls_priority}")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> controls_az = {controls_az}")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> controls_el = {controls_el}")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> log_array_sizes = {log_array_sizes}\n")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> controls_start = {controls_start}\n")
logger.info(f"test_scan_controller_w_scheduler:computation_variables -> controls_start = {controls_end}\n")

controls = []
controls_plt = []
for i in range(len(controls_period)):
    t = np.arange(controls_start[i], controls_end[i], controls_period[i])
    logger.info(f"test_scan_controller_w_scheduler -> generating time points - size={len(t)}")
    controls.append(scanner_ctrl.generate_controls(t, eiscat3d, scans[i], scheduler=scheduler, priority=controls_priority[i]))
    controls_plt.append(scanner_ctrl.generate_controls(t, eiscat3d, scans[i], scheduler=scheduler, priority=controls_priority[i]))



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

p.start('object_initialization')

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

p.stop('object_initialization')
logger.info("test_tracker_controller -> object created :")
logger.info(f"test_tracker_controller -> {space_object}")

logger.info("test_tracker_controller -> sampling equidistant states on the orbit")


# create state time array
p.start('equidistant_sampling')
t_states = equidistant_sampling(
    orbit = space_object.state, 
    start_t = 0, 
    end_t = end_t, 
    max_dpos=50e3,
)
p.stop('equidistant_sampling')

# get object states in ECEF frame
p.start('get_state')
object_states = space_object.get_state(t_states)
p.stop('get_state')

logger.info("test_tracker_controller -> object states computation done ! ")
logger.info(f"test_tracker_controller -> t_states -> {t_states.shape}")

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
p.start('find_simultaneous_passes')
eiscat_passes = find_simultaneous_passes(t_states, object_states, [*eiscat3d.tx, *eiscat3d.rx])
p.stop('find_simultaneous_passes')
logger.info(f"test_tracker_controller -> Passes : eiscat_passes={eiscat_passes}")

p.start('generate_tracking_controls')
tracker_controller = controllers.Tracker(logger=logger, profiler=p)

for pass_id in range(np.shape(eiscat_passes)[0]):
    logger.info(f"test_tracker_controller -> Computing tracking controls for pass {pass_id}:")

    tracking_states = object_states[:, eiscat_passes[pass_id].inds]
    t_states_i = t_states[eiscat_passes[pass_id].inds]

    ax.plot(tracking_states[0], tracking_states[1], tracking_states[2], "-b")
    
    p.start('intitialize_controller')
    t_controller = np.arange(t_states_i[0], t_states_i[-1]+tracking_period, tracking_period)

    p.stop('intitialize_controller')

    controls.append(tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, scheduler=scheduler, priority=0, states_per_slice=20, interpolator=interpolation.Linear))
    controls_plt.append(tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, scheduler=scheduler, priority=0, states_per_slice=20, interpolator=interpolation.Linear))
    
    logger.info("test_tracker_controller -> Controls generated")
p.stop('generate_tracking_controls')

print("generate_tracking_controls")

final_control_sequence = scheduler.run(controls)

p.stop("test_scan_controller_w_scheduler:compute_controls")

p.start("test_scan_controller_w_scheduler:get_radar_states")
radar_states = eiscat3d.control(final_control_sequence)
p.stop("test_scan_controller_w_scheduler:get_radar_states")


for txi in range(len(eiscat3d.tx)):
    for rxi in range(len(eiscat3d.rx)):
        print(f"{txi}, {rxi}")
        p.start("test_scan_controller_w_scheduler:get_dir_count:loop")
        print("loop -> ", get_dir_count_loop(radar_states["pointing_direction"][0]["rx"][rxi, txi]))    
        p.stop("test_scan_controller_w_scheduler:get_dir_count:loop")

        p.start("test_scan_controller_w_scheduler:get_dir_count:vec")
        print("vec -> ", get_dir_count_vec(radar_states["pointing_direction"][0]["rx"][rxi, txi]))
        p.stop("test_scan_controller_w_scheduler:get_dir_count:vec")

        p.start("test_scan_controller_w_scheduler:get_dir_count:c")
        print("c -> ", get_dir_count_c(radar_states["pointing_direction"][0]["rx"][rxi, txi]))
        p.stop("test_scan_controller_w_scheduler:get_dir_count:c")

print(p)