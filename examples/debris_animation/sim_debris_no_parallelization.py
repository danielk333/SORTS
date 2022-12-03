#!/usr/bin/env python

'''
TSDR beam park simulation
===========================
'''
import pathlib
import configparser

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle

import sorts
import pyant

# radar = sorts.radars.tsdr
radar = sorts.radars.eiscat3d_interp

from sorts import SpaceObject, Simulation
from sorts import MPI_single_process, MPI_action, iterable_step, store_step, cached_step, iterable_cache
from sorts.radar.scans import Beampark
from sorts.targets.population import master_catalog, master_catalog_factor
from sorts import interpolation

from sorts import plotting

# gets example config 
master_path = "/home/thomas/venvs/sorts/master catalog/celn_20090501_00.sim"
simulation_root = "./results/"

# initializes the propagator
from sorts.targets.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)

# simulation time
end_t = 3600.0*24

population = master_catalog(
    master_path,
    propagator = Prop_cls,
    propagator_options = Prop_opts,
)
population = master_catalog_factor(population, treshhold = 0.1)
population.delete(slice(10000, None, None)) # DELETE ALL BUT THE FIRST 10000
inds = list(range(len(population)))

# plotting parameters
duration = 240 # seconds
fps = 30

t_anim = np.linspace(0, end_t, int(duration*fps)+1)
object_states = np.ndarray((len(inds), 3, len(t_anim)))

for i in range(len(inds)):
    print("propagating object id ", i, "/", len(inds))

    obj = population.get_object(inds[i])
    object_states[i, :, :] = obj.get_state(t_anim)[0:3]


for i in range(37, len(t_anim)):
    print("plot ", i, "/", len(t_anim))

    fig = plt.figure(dpi=600)
    ax = fig.add_subplot(111, projection='3d')

    plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

    ax.plot(object_states[:, 0, i], object_states[:, 1, i], object_states[:, 2, i], "o", color="darkblue", ms=0.1)

    ax.set_xlim([-40e6, 40e6])
    ax.set_ylim([-40e6, 40e6])
    ax.set_zlim([-40e6, 40e6])

    fig.savefig(f"./imgs/debris_img_{i:05d}.jpg")   # save the figure to file
    plt.close()
