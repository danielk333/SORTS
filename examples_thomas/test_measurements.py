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

from matplotlib.ticker import AutoMinorLocator

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
end_t = 3600*24

# scheduler properties
t0 = 0
scheduler_period = 3600 # [s]

logger.info("test_scan_controller_w_scheduler -> scheduler initialized:")
logger.info(f"test_scan_controller_w_scheduler:scheduler_variables -> t0 = {t0}")
logger.info(f"test_scan_controller_w_scheduler:scheduler_variables -> scheduler_period = {scheduler_period}\n")

# RADAR definition
eiscat3d = instances.eiscat3d_interp
eiscat3d.logger = logger
eiscat3d.profiler = p

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
scan = Fence(azimuth=0, min_elevation=75, dwell=0.5, num=100)

t = np.arange(0, end_t, 0.5)
controls = scanner_ctrl.generate_controls(t, eiscat3d, scan, scheduler=scheduler)


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

p.start('object_initialization')
# Creating space object
# Object properties
orbits_a = np.array([7200, 7250, 7080, 7000])*1e3 # km
orbits_i = np.array([80, 82, 85, 79]) # deg
orbits_raan = np.array([86, 160, 180, 90]) # deg
orbits_aop = np.array([0, 50, 40, 55]) # deg
orbits_mu0 = np.array([60, 5, 30, 8]) # deg

space_objects = []
for obj_id in range(len(orbits_a)):
    space_objects.append(space_object.SpaceObject(
            Prop_cls,
            propagator_options = Prop_opts,
            a = orbits_a[obj_id], 
            e = 0.01,
            i = orbits_i[obj_id],
            raan = orbits_raan[obj_id],
            aop = orbits_aop[obj_id],
            mu0 = orbits_mu0[obj_id],
            
            epoch = 53005.0,
            parameters = dict(
                d = 0.1,
            ),
        ))
p.stop('object_initialization')

logger.info("computing radar states")

p.start("test_scan_controller_w_scheduler:get_radar_states")
radar_states = eiscat3d.control(controls)
p.stop("test_scan_controller_w_scheduler:get_radar_states")

print("radar_states ", radar_states.t[0])
logger.info("computing measurements")

fig = plt.figure()
axes = fig.subplots(3, 1, sharex=True)

colors = ["b", "r", "c", "k"]
for oid in range(len(space_objects)):
    p.start("test_scan_controller_w_scheduler:measurement")
    data = eiscat3d.compute_measurements(radar_states, space_objects[oid], logger=logger, profiler=p)
    p.stop("test_scan_controller_w_scheduler:measurement")

    logger.info(f"plotting SNR measurements for object {space_object}")
    for period_id in range(len(data["t"])):
        for i in range(len(eiscat3d.rx)):
            axes[i].plot(data["t"][period_id][0, i], 10*np.log10(data["snr"][period_id][0, i]), "-" + colors[oid])

            axes[i].set_xlabel(r"$t$ [$s$]")
            axes[i].set_ylabel("$10 log(P_{s}/P_{noise})$ [$dB$]")

            axes[i].grid()

            axes[i].tick_params(which="minor", direction="in", bottom=True, top=True, left=True, right=True)
            axes[i].tick_params(which="major", direction="in", bottom=True, top=True, left=True, right=True)

            axes[i].xaxis.set_minor_locator(AutoMinorLocator())
            axes[i].yaxis.set_minor_locator(AutoMinorLocator())

            plt.subplots_adjust(wspace=0, hspace=0)

plt.show()

print(p)