#!/usr/bin/env python

'''
TSDR beam park simulation
===========================
'''
import pathlib
import configparser

import numpy as np
import matplotlib.pyplot as plt

import os

from sorts import SpaceObject, Simulation
from sorts import MPI_single_process, MPI_action, iterable_step, store_step, cached_step, iterable_cache
from sorts.targets.population import master_catalog, master_catalog_factor
from sorts import equidistant_sampling

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

def count(result, item):
    if result is None:
        result = 1
    else:
        result += 1
    return result

def _sum(result, item):
    if result is None:
        result = item
    else:
        result += item
    return result


class ScanningSimulation(Simulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.population = master_catalog(
            master_path,
            propagator = Prop_cls,
            propagator_options = Prop_opts,
        )
        self.population = master_catalog_factor(self.population, treshhold=0.1)
        self.population.delete(slice(10000, None, None)) # DELETE ALL BUT THE FIRST 10000
        self.inds = list(range(len(self.population)))

        # plotting parameters
        duration = 60 # seconds
        fps = 30

        self.t_anim = np.linspace(0, end_t, int(duration*fps)+1)
        self.so_states = np.ndarray((len(self.population), 3, len(self.t_anim)), dtype=float)

        if self.logger is not None:
            self.logger.always(f'Population of size {len(self.population)} objects loaded.')

        self.steps['propagate'] = self.get_states
        self.steps['interpolate'] = self.interpolate_states
        self.steps['plot'] = self.generate_plots

    @MPI_action(action='barrier')
    @store_step(store='object_prop')
    @iterable_step(iterable='inds', MPI=True, log=True, reduce=count)
    @cached_step(caches='h5')
    def get_states(self, index, item, **kwargs):
        obj = self.population.get_object(item)

        t = equidistant_sampling(
            orbit = obj.orbit,
            start_t = self.t_anim[0],
            end_t = self.t_anim[-1] + np.diff(self.t_anim)[-1],
            max_dpos=50e3,
        )
        state = obj.get_state(t)
        return state, t

    @iterable_cache(steps='propagate', caches='h5', MPI=False, log=True, reduce=lambda t,x: None)
    def interpolate_states(self, index, item, **kwargs):
        state, t = item
        self.so_states[index, :, :] = interpolation.Linear(state, t).get_state(self.t_anim)[0:3, :]        

    @MPI_action(action='barrier')
    @iterable_step(iterable='t_anim', MPI=True, log=True, reduce=lambda t,x: None)
    def generate_plots(self, index, item, **kwargs):
        fig = plt.figure(dpi=600)
        ax = fig.add_subplot(111, projection='3d')

        plotting.grid_earth(ax, num_lat=25, num_lon=50, alpha=0.1, res = 100, color='black', hide_ax=True)

        ax.plot(self.so_states[:, 0, index], self.so_states[:, 1, index], self.so_states[:, 2, index], "o", color="red", ms=1)

        ax.set_xlim([-40e6, 40e6])
        ax.set_ylim([-40e6, 40e6])
        ax.set_zlim([-40e6, 40e6])

        fig.savefig(f"./imgs/debris_img_{index:05d}.jpg")   # save the figure to file
        plt.close()

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

os.system("cd ./imgs")
os.system("fmpeg -framerate 30 -pattern_type glob -i '*.jpg' -c:v libx264 -pix_fmt yuv420p out.mp4")
os.system("rm -rf *.jpg")
os.system("cd ..")