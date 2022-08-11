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
radar = sorts.radars.tsdr_fence

from sorts import SpaceObject, Simulation
from sorts import MPI_single_process, MPI_action, iterable_step, store_step, cached_step, iterable_cache
from sorts.radar.scans import Beampark
from sorts.targets.population import master_catalog, master_catalog_factor

# gets example config 
#try:
#     base_pth = pathlib.Path(__file__).parents[1].resolve()
# except NameError:
#     base_pth = pathlib.Path('.').parents[1].resolve()

# config = configparser.ConfigParser(interpolation=None)
# config.read([base_pth / 'example_config.conf'])
# master_path = pathlib.Path(config.get('TSDR_beampark__ngl.py', 'master_catalog'))
# simulation_root = pathlib.Path(config.get('TSDR_beampark__ngl.py', 'simulation_root'))
# if not simulation_root.is_absolute():
#     simulation_root = base_pth / simulation_root.relative_to('.')
# if not master_path.is_absolute():
#     master_path = base_pth / master_path.relative_to('.')

simulation_root = '.'

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
scan = Beampark(azimuth=radar.tx[0].beam.azimuth, elevation=radar.tx[0].beam.elevation)

# initializes the simulation from the master catalog
# pop = master_catalog(
#     master_path,
#     propagator = Prop_cls,
#     propagator_options = Prop_opts,
# )
# pop = master_catalog_factor(pop, treshhold = 0.1)
# pop.delete(slice(100, None, None)) #DELETE ALL BUT THE FIRST 100

pop = sorts.Population(
    fields = ['oid','a','e','i','raan','aop','mu0','mjd0', 'm', 'A', 'C_R', 'C_D'],
    space_object_fields = ['oid', 'm', 'A', 'C_R', 'C_D'],
    state_fields = ['a','e','i','raan','aop','mu0'],
    epoch_field = {'field': 'mjd0', 'format': 'mjd', 'scale': 'utc'},
    propagator = SGP4,
    propagator_options = dict(
        settings = dict(
            out_frame='TEME',
        ),
    )
)
pop.allocate(1) # allocate 100 space objects

# create space objects
# * 0: oid - Object ID
# * 1: a - Semi-major axis in m
# * 2: e - Eccentricity 
# * 3: i - Inclination in degrees
# * 4: raan - Right Ascension of ascending node in degrees
# * 5: aop - Argument of perihelion in degrees
# * 6: mu0 - Mean anoamly in degrees
# * 7: mjd0 - Epoch of object given in Modified Julian Date
pop['oid'] = 0
pop['a'] = 7200e3
pop['e'] = 0.1
pop['i'] = 75
pop['raan'] = 79
pop['aop'] = 0
pop['mu0'] = 62
pop['mjd0'] = 57125.7729
pop['m'] = 0.1
pop['A'] = 1
pop['C_R'] = 0.1
pop['C_D'] = 2.7

# create scanner controller and generate radar controls
scanner_controller = sorts.Scanner()
t = np.arange(0, end_t, scan.dwell())
controls = scanner_controller.generate_controls(t, radar, scan, max_points=1000)

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
    def __init__(self, population, radar_states, *args, **kwargs):
        self.population = population
        self.radar_states = radar_states
        self.inds = list(range(len(population)))

        self.ipp_detection_lim = 1

        super().__init__(*args, **kwargs)

        if self.logger is not None:
            self.logger.always(f'Population of size {len(population)} objects loaded.')

        self.steps['propagate'] = self.get_states
        self.steps['passes'] = self.find_passes
        self.steps['observe'] = self.observe_passes
        self.steps['detected'] = self.count_detected

    @MPI_action(action='barrier')
    @store_step(store='object_prop')
    @iterable_step(iterable='inds', MPI=True, log=True, reduce=count)
    @cached_step(caches='h5')
    def get_states(self, index, item):
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
    def find_passes(self, index, item):
        state, t = item
        passes = radar.find_passes(t, state, cache_data=False)
        return passes

    @MPI_action(action='barrier')
    @iterable_cache(steps='passes', caches='pickle', MPI=True, log=True, reduce=lambda t,x: None)
    @cached_step(caches='pickle')
    def observe_passes(self, index, item):
        # TODO understand role of item : t, x ? states
        data = radar.observe_passes(item, self.radar_states, self.population.get_object(index), snr_limit=False, parallelization=False)
        return data


    @store_step(store='detected_objects')
    @MPI_action(action='gather', root=0)
    @iterable_cache(steps='observe', caches='pickle', MPI=True, log=True, reduce=_sum)
    def count_detected(self, index, item):
        detected_ = 0.0
        for pass_ in item:
            for tx_ps in pass_:
                for txrx_ps in tx_ps:
                    if np.sum(np.log10(txrx_ps['snr'])*10.0 > 10.0) >= self.ipp_detection_lim:
                        detected_ = 1.0
        return detected_


    @MPI_single_process(process_id = 0)
    def total_detected(self):
        return np.sum(self.detected_objects)


    @MPI_single_process(process_id = 0)
    def plot(self):
        pass





sim = ScanningSimulation(
    population = pop,
    radar_states = radar_states,
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


