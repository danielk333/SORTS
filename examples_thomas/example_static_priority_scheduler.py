#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 08:16:43 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

import sorts


# scheduler and Computation properties
t0                  = 0
scheduler_period    = 50 # [s] -> 2 minutes - can go up to 10mins or more depending on the available RAM
end_t               = 100

# RADAR definition
eiscat3d = sorts.radars.eiscat3d

# ======================================= Scheduler ========================================

scheduler = sorts.StaticPriorityScheduler(eiscat3d, t0, scheduler_period)

# ======================================== Controls =========================================
# ---------------------------------------- Scanner -----------------------------------------


# controls parameters 
controls_period     = np.array([1, 2, 2])
controls_t_slice    = np.array([0.5, 1, 1.5])
controls_priority   = np.array([3, 2, 1], dtype=int) # control priorities -> 1, 2, 3
controls_start      = np.array([0, 4, 49.5], dtype=float)
controls_end        = np.array([end_t, end_t, end_t], dtype=float)
controls_az         = np.array([90, 45, 80])
controls_el         = np.array([10, 20, 30])

scans = []
for i in range(len(controls_period)):
    scans.append(sorts.scans.Fence(azimuth=controls_az[i], min_elevation=controls_el[i], dwell=controls_t_slice[i], num=int(controls_end[i]/controls_t_slice[i])))

# Generate scanning controls
controls        = []
scanner_ctrl    = sorts.Scanner()
for i in range(len(controls_period)):
    t = np.arange(controls_start[i], controls_end[i], controls_period[i])
    controls.append(scanner_ctrl.generate_controls(t, eiscat3d, scans[i], scheduler=scheduler, priority=controls_priority[i]))

# TRACKER
# Object definition
# Propagator
Prop_cls = sorts.propagator.Kepler
Prop_opts = dict(
    settings = dict(
        out_frame='ITRS',
        in_frame='TEME',
    ),
)


# ---------------------------------------- Tracker -----------------------------------------

# Creating space object
space_object = sorts.SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = 7200e3, 
        e = 0.1,
        i = 80.0,
        raan = 86.0,
        aop = 0.0,
        mu0 = 60.0,
        
        epoch = 53005.0,
        parameters = dict(
            d = 0.1,
        ),
    )

# create state time array
t_states = sorts.equidistant_sampling(
    orbit=space_object.state, 
    start_t=0, 
    end_t=end_t, 
    max_dpos=50e3,
)

# get object states in ECEF frame and passes
object_states   = space_object.get_state(t_states)
eiscat_passes   = sorts.find_simultaneous_passes(t_states, object_states, [*eiscat3d.tx, *eiscat3d.rx])

# Tracker controller parameters
tracking_period = 15
t_slice         = 10

# create Tracker controller with highest priority (prio=0)
tracker_controller = sorts.Tracker()

for pass_id in range(np.shape(eiscat_passes)[0]):
    tracking_states = object_states[:, eiscat_passes[pass_id].inds]
    t_states_i      = t_states[eiscat_passes[pass_id].inds]
    t_controller    = np.arange(t_states_i[0], t_states_i[-1]+tracking_period, tracking_period)
    controls.append(tracker_controller.generate_controls(t_controller, eiscat3d, t_states_i, tracking_states, t_slice=t_slice, scheduler=scheduler, priority=0, states_per_slice=10, interpolator=sorts.interpolation.Linear))
    
# ==================================== Run scheduler ======================================


final_control_sequence = scheduler.run(controls, t_start=29, t_end=70)


# ======================================= Plotting ========================================

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plotting station ECEF positions and grid earth
sorts.plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)
for tx in eiscat3d.tx:
    ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in eiscat3d.rx:
    ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

# plot passes
for pass_id in range(np.shape(eiscat_passes)[0]):
    tracking_states = object_states[:, eiscat_passes[pass_id].inds]
    ax.plot(tracking_states[0], tracking_states[1], tracking_states[2], "-b")

# plot scheduler schedule
figs = sorts.plotting.plot_scheduler_control_sequence(controls, final_control_sequence, scheduler)

# plot control sequences pointing directions
fmts = ["b-", "m-", "k-", "-c"]
for period_id in range(final_control_sequence.n_periods):
    for ctrl_i in range(len(controls)):
        ax = sorts.plotting.plot_beam_directions(controls[ctrl_i].get_pdirs(period_id), eiscat3d, ax=ax, fmt=fmts[ctrl_i], linewidth_rx=0.08, linewidth_tx=0.08, alpha=0.001)

# plot scheduler pointing directions
for period_id in range(final_control_sequence.n_periods):
    ax = sorts.plotting.plot_beam_directions(final_control_sequence.get_pdirs(period_id), eiscat3d, ax=ax)

plt.show()