#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 08:16:43 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['agg.path.chunksize'] = 100000000

from sorts import Simulation
from sorts import MPI_single_process, MPI_action, iterable_step, store_step, cached_step, iterable_cache
from sorts.radar.system import instances
from sorts import SpaceObjectTracker
from sorts.radar import controllers

from sorts.common import profiling
from sorts import plotting
from sorts import space_object

from sorts.targets.propagator import Kepler
from sorts import find_simultaneous_passes, equidistant_sampling
from sorts import interpolation

import os.path
import pickle  

# gets example config 
simulation_root = "./results/"

# initializes the propagator
from sorts.targets.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)

class ScanningSimulation(Simulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # RADAR definition
        self.radar = instances.eiscat3d

        # controller definition
        self.tracking_period = 50
        self.t_slice = 2
        self.t_start = 0
        self.t_end = 3600*24
        self.epoch = 53005.0

        # control time array
        self.t_tracking = np.arange(self.t_start, self.t_end, self.tracking_period)

        # multiple space object tracker controller
        self.so_tracking_controller = SpaceObjectTracker(logger=None, profiler=None)

        # Propagator
        Prop_cls = Kepler
        Prop_opts = dict(
            settings = dict(
                out_frame='ITRS',
                in_frame='TEME',
            ),
        )

        # Space object orbital parameters
        orbits_a = np.array([7200, 7200, 8500, 12000, 10000])*1e3 # m
        orbits_i = np.array([80, 80, 105, 105, 80]) # deg
        orbits_raan = np.array([86, 86, 160, 180, 90]) # deg
        orbits_aop = np.array([0, 0, 50, 40, 55]) # deg
        orbits_mu0 = np.array([2, -10, 5, 85, 18]) # deg
        orbits_e = np.array([0.1, 0.05, 0.3, 0.02, 0.4]) # deg

        # space object priorities
        priorities = np.array([4, 3, 1, 2, 5])

        # space object instances
        self.space_objects = []
        for so_id in range(len(orbits_a)):
            self.space_objects.append(space_object.SpaceObject(
                    Prop_cls,
                    propagator_options = Prop_opts,
                    a = orbits_a[so_id], 
                    e = 0.1,
                    i = orbits_i[so_id],
                    raan = orbits_raan[so_id],
                    aop = orbits_aop[so_id],
                    mu0 = orbits_mu0[so_id],
                    
                    epoch = self.epoch,
                    parameters = dict(
                        d = 0.1,
                    ),
                ))

        # generate/cache controls
        if not os.path.exists("./controls.pickle"):
            self.controls = self.so_tracking_controller.generate_controls(self.t_tracking, self.radar, self.space_objects, self.epoch, self.t_slice, space_object_priorities=priorities, save_states=True)
            with open("./controls.pickle", 'wb') as handle:
                pickle.dump(self.controls, handle, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open("./controls.pickle", 'rb') as handle:
                self.controls = pickle.load(handle)

        # animation properties
        self.duration = 60
        self.fps = 30
        self.t_anim = np.linspace(self.t_start, self.t_end, int(self.duration*self.fps)+1)

        # interpolate space object states to each animation time step
        self.states_interp = np.ndarray((len(self.space_objects), 3, len(self.t_anim)))
        for space_object_index in range(len(self.space_objects)):
            self.states_interp[space_object_index, :, :] = interpolation.Linear(self.controls.meta["objects_states"][space_object_index], self.t_tracking).get_state(self.t_anim)[0:3, :]

        # simulation steps
        self.steps['plot'] = self.generate_plots
        self.steps['video'] = self.generate_video

    @MPI_action(action='barrier')
    @iterable_step(iterable='t_anim', MPI=True, log=True, reduce=lambda t,x: None)
    def generate_plots(self, index, item, **kwargs):
        colors = ['red','orange','yellow','green','blue','purple']

        # skip step if image already exists
        if os.path.exists(f"./imgs/debris_img_{index:05d}.jpg"):
            return None

        # plotting results
        fig = plt.figure(dpi=800)
        ax = fig.add_subplot(111, projection='3d')

        # Plotting station ECEF positions
        plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

        # Plotting station ECEF positions
        for tx in self.radar.tx:
            ax.plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or', ms=1)
        for rx in self.radar.rx:
            ax.plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og', ms=1)       

        # plot all space object states
        for so_id in range(len(self.space_objects)):
            ax.plot(self.states_interp[so_id, 0, 0:index+1], self.states_interp[so_id, 1, 0:index+1], self.states_interp[so_id, 2, 0:index+1], '-', alpha=0.1, color="darkblue", linewidth=1)
            ax.plot(self.states_interp[so_id, 0, index], self.states_interp[so_id, 1, index], self.states_interp[so_id, 2, index], 'o', color="darkblue", alpha=1, ms=1)
            ax.text(self.states_interp[so_id, 0, index]+2000, self.states_interp[so_id, 1, index]+2000, self.states_interp[so_id, 2, index]+2000,  '%s' % (str(so_id)), size=5, zorder=1)
            

        # plot tracking states
        ecef_tracking = self.controls.meta["tracking_states"]
        object_ids = self.controls.meta["state_priorities"]

        # plot all pointing directions before time step
        for period_id in range(self.controls.n_periods):

            # skip search if the iterator is not within the correct period id
            if self.controls.t[period_id][-1] < self.t_anim[index]:
                pdirs = self.controls.get_pdirs(period_id)
                ax = plotting.plot_beam_directions(pdirs, self.radar, ax=ax, zoom_level=0.6, azimuth=30, elevation=20, alpha=0.1)
                
                mask = np.logical_or(np.abs(pdirs["t"][1:] - pdirs["t"][:-1]) > self.tracking_period, object_ids[period_id][1:] - object_ids[period_id][:-1] != 0)
                transition_ids = np.where(mask)[0]+1

                for i in range(len(transition_ids)+1):
                    if i == 0:
                        i_start = 0
                    else:
                        i_start = transition_ids[i-1]

                    if i == len(transition_ids):
                        i_end = len(self.t_tracking)+1
                    else:
                        i_end = transition_ids[i]

                    ax.plot(ecef_tracking[period_id][0, i_start:i_end], ecef_tracking[period_id][1, i_start:i_end], ecef_tracking[period_id][2, i_start:i_end], '-r')

            else:
                mask = self.controls.t[period_id] <= self.t_anim[index]
                controls_t = self.controls.t[period_id][mask]
                object_ids_i = object_ids[period_id][mask]

                if len(controls_t) > 0:
                    pdirs_last = self.controls.get_pdirs(period_id).copy()

                    pdirs_last["t"] = pdirs_last["t"][mask]
                    pdirs_last["tx"] = pdirs_last["tx"][:, :, :, mask]
                    pdirs_last["rx"] = pdirs_last["rx"][:, :, :, mask]

                    ax = plotting.plot_beam_directions(pdirs_last, self.radar, ax=ax, zoom_level=0.6, azimuth=30, elevation=20, alpha=0.1)

                    pdirs_last["t"] = np.array(pdirs_last["t"][-1])[None]
                    pdirs_last["tx"] = np.array(pdirs_last["tx"][:, :, :, -1][:, :, :, None])
                    pdirs_last["rx"] = np.array(pdirs_last["rx"][:, :, :, -1][:, :, :, None])

                    if self.t_anim[index] < pdirs_last["t"] + self.tracking_period:
                        ax = plotting.plot_beam_directions(pdirs_last, self.radar, ax=ax, zoom_level=0.6, azimuth=30, elevation=20)

                    mask = np.logical_or(np.abs(controls_t[1:] - controls_t[:-1]) > self.tracking_period, object_ids_i[1:] - object_ids_i[:-1] != 0)
                    transition_ids = np.where(mask)[0]+1

                    for i in range(len(transition_ids)+1):
                        if i == 0:
                            i_start = 0
                        else:
                            i_start = transition_ids[i-1]

                        if i == len(transition_ids):
                            i_end = len(controls_t)
                        else:
                            i_end = transition_ids[i]

                        ax.plot(ecef_tracking[period_id][0, i_start:i_end], ecef_tracking[period_id][1, i_start:i_end], ecef_tracking[period_id][2, i_start:i_end], '-r')
                break

        ax.view_init(20, self.t_anim[index]*360/self.t_end*4)

        ax.set_xlim([-10e6, 10e6])
        ax.set_ylim([-10e6, 10e6])
        ax.set_zlim([-10e6, 10e6])
        
        fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

        fig.savefig(f"./imgs/debris_img_{index:05d}.jpg")   # save the figure to file
        plt.close()

    @MPI_action(action='barrier')
    @MPI_single_process(process_id = 0)
    def generate_video(self):
        del self.controls
        del self.states_interp 

        os.system(f"ffmpeg -framerate {self.fps} -pattern_type glob -i './imgs/*.jpg' -c:v libx264 -pix_fmt yuv420p out.mp4")

sim = ScanningSimulation(
    scheduler = None,
    root = simulation_root,
    logger=True,
    profiler=True,
)

sim.profiler.start('total')
sim.run()

sim.profiler.stop('total')
sim.logger.always('\n' + sim.profiler.fmt(normalize='total'))

print("DO NOT FORGET TO REMOVE ALL IMAGES IN ./imgs/")