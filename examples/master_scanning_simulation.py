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
end_t = 600.0
# create Beampark scan 
scan = Beampark(azimuth=radar.tx[0].beam.azimuth, elevation=radar.tx[0].beam.elevation, dwell=0.1)

# create scanner controller and generate radar controls
scanner_controller = sorts.Scanner()
t = np.arange(0, end_t, scan.dwell())
controls = scanner_controller.generate_controls(t, radar, scan, max_points=1000, cache_pdirs=False)

# generate radar states from the radar controls
radar_states = radar.control(controls)

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


class ScanningSimulation(sorts.Simulation):
    def __init__(self, radar_states, *args, **kwargs):
        self.ipp_detection_lim = 1
        self.radar_states = radar_states

        super().__init__(*args, **kwargs)

        self.population = master_catalog(
            master_path,
            propagator = Prop_cls,
            propagator_options = Prop_opts,
        )
        self.population = master_catalog_factor(self.population, treshhold = 0.1)
        self.population.delete(slice(100, None, None)) # DELETE ALL BUT THE FIRST 100

        self.inds = list(range(len(self.population)))

        if self.logger is not None:
            self.logger.always(f'Population of size {len(self.population)} objects loaded.')

        self.orbital_parameters = []

        self.steps['propagate'] = self.get_states
        self.steps['passes'] = self.find_passes
        self.steps['observe'] = self.observe_passes
        self.steps['detected'] = self.count_detected

    @MPI_action(action='barrier')
    @store_step(store='object_prop')
    @iterable_step(iterable='inds', MPI=True, log=True, reduce=count)
    @cached_step(caches='h5')
    def get_states(self, index, item, **kwargs):
        obj = self.population.get_object(item)

        t = sorts.equidistant_sampling(
            orbit = obj.orbit,
            start_t = self.radar_states.t[0][0],
            end_t = self.radar_states.t[-1][-1] + (self.radar_states.t[-1][-1] - self.radar_states.t[-1][-2]),
            max_dpos=1e3,
        )
        state = obj.get_state(t)
        return state, t

    @MPI_action(action='barrier')
    @iterable_cache(steps='propagate', caches='h5', MPI=True, log=True, reduce=lambda t,x: None)
    @cached_step(caches='pickle')
    def find_passes(self, index, item, **kwargs):
        state, t = item
        passes = radar.find_passes(t, state, cache_data=False)
        return passes

    @MPI_action(action='barrier')
    @iterable_cache(steps='passes', caches='pickle', MPI=True, log=True, reduce=lambda t,x: None)
    @cached_step(caches='pickle')
    def observe_passes(self, index, item, **kwargs):
        # TODO understand role of item : t, x ? states
        if self.logger is not None:
            self.logger.always(f'Observing object {index}')

        data = radar.observe_passes(item, self.radar_states, self.population.get_object(index), snr_limit=False, parallelization=True)
        
        if self.logger is not None:
            self.logger.always(f'Object {index} observation done')

        return data


    # @store_step(store='detected_objects')
    # @MPI_action(action='gather', root=0)
    # @iterable_cache(steps='observe', caches='pickle', MPI=True, log=_sum)
    # def count_detected(self, index, item, **kwargs):
    #     detected_ = 0.0
    #     for pass_ in item:
    #         for tx_ps in pass_:
    #             for txrx_ps in tx_ps:
    #                 data = sorts.radar.measurement.get_max_snr_measurements(txrx_ps, copy=True)
    #                 if np.sum(np.log10(data["measurements"]["snr"])*10.0 > 10.0) >= self.ipp_detection_lim:
    #                     detected_ = 1.0

    #                     # obj_a = self.population.get_object(index).orbit.a[0]
    #                     # obj_e = self.population.get_object(index).orbit.e[0]
    #                     # obj_i = self.population.get_object(index).orbit.i[0]
    #                     # obj_aop = self.population.get_object(index).orbit.omega[0]
    #                     # obj_raan = self.population.get_object(index).orbit.Omega[0]

    #                     # self.orbital_parameters.append([obj_a, obj_e, obj_i, obj_aop, obj_raan])

    #     return detected_


    @store_step(store='detected_objects')
    @MPI_action(action='gather', root=0)
    @iterable_cache(steps='observe', caches='pickle', MPI=True, log=True, reduce=_sum)
    def count_detected(self, index, item, **kwargs):
        detected_ = 0.0
        for pass_ in item:
            for tx_ps in pass_:
                for txrx_ps in tx_ps:
                    data = sorts.radar.measurement.get_max_snr_measurements(txrx_ps, copy=True)
                    if np.sum(np.log10(data["measurements"]["snr"])*10.0 > 10.0) >= self.ipp_detection_lim:
                        detected_ = 1.0

        return detected_


    @MPI_single_process(process_id = 0)
    def total_detected(self):
        return np.sum(self.detected_objects)

    @MPI_single_process(process_id = 0)
    def plot(self):
        pass
        # plt_vars = ["a", "e", "i", "aop", "raan"]

        # fig = plt.figure()
        # axes = fig.subplots(len(plt_vars)-1, len(plt_vars)-1)

        # print(self.orbital_parameters)
        # self.orbital_parameters = np.asfarray(self.orbital_parameters).T

        # pop = master_catalog(
        #     master_path,
        #     propagator = Prop_cls,
        #     propagator_options = Prop_opts,
        # )

        # for j in range(0, len(plt_vars)-1):
        #     for i in range(j+1, len(plt_vars)):
        #         ix = i
        #         iy = j

        #         axes[i-1, j].scatter(pop.data[plt_vars[ix]], pop.data[plt_vars[iy]], marker="+", s=10, c="b")
        #         axes[i-1, j].scatter(self.orbital_parameters[ix], self.orbital_parameters[iy], marker="+", s=10, c="r")
        #         axes[i-1, j].set_xlabel(plt_vars[ix])
        #         axes[i-1, j].set_ylabel(plt_vars[iy])
        #         axes[i-1, j].grid()




sim = ScanningSimulation(
    radar_states=radar_states,
    scheduler = None,
    root = simulation_root,
    logger=True,
    profiler=True,
)
# sim.delete('debug')
# sim.branch('debug')

# sim.ipp_detection_lim = 10

sim.profiler.start('total')
sim.run()

print(f'Propagations: {sim.object_prop}')

sim.profiler.stop('total')
sim.logger.always('\n' + sim.profiler.fmt(normalize='total'))

print(f'Total detected: {sim.total_detected()}')

sim.plot()
plt.show()


## EX

# sim.run('propagate')
# sim.run('passes')
# for ind, freq in enumerate([1.2e6, 2.4e6]):
#     sim.checkout('master')
#     sim.branch(f'f{ind}')
#     sim.radar.tx[0].beam.frequency = freq
#     sim.radar.rx[0].beam.frequency = freq
#     sim.radar_states = sim.radar.control(controls)
#     sim.run('observe')

##


