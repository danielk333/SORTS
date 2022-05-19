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

from sorts.radar.scheduler import StaticList, ObservedParameters
from sorts.radar.controllers import Scanner
from sorts import SpaceObject, Simulation
from sorts import MPI_single_process, MPI_action, iterable_step, store_step, cached_step, iterable_cache
from sorts.radar.scans import Beampark
from sorts.targets.population import master_catalog, master_catalog_factor

from sorts.targets.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)

# pyant.plotting.gain_heatmap(radar.tx[0].beam, resolution=300, min_elevation=80.0)
# plt.show()


end_t = 600.0
scan = Beampark(azimuth=radar.tx[0].beam.azimuth, elevation=radar.tx[0].beam.elevation)

try:
    base_pth = pathlib.Path(__file__).parents[1].resolve()
except NameError:
    base_pth = pathlib.Path('.').parents[1].resolve()

config = configparser.ConfigParser(interpolation=None)
config.read([base_pth / 'example_config.conf'])
master_path = pathlib.Path(config.get('TSDR_beampark__ngl.py', 'master_catalog'))

simulation_root = pathlib.Path(config.get('TSDR_beampark__ngl.py', 'simulation_root'))

if not simulation_root.is_absolute():
    simulation_root = base_pth / simulation_root.relative_to('.')


if not master_path.is_absolute():
    master_path = base_pth / master_path.relative_to('.')

pop = master_catalog(
    master_path,
    propagator = Prop_cls,
    propagator_options = Prop_opts,
)
pop = master_catalog_factor(pop, treshhold = 0.1)

pop.delete(slice(100,None,None)) #DELETE ALL BUT THE FIRST 100

class ObservedScanning(StaticList, ObservedParameters):
    pass

scan_sched = Scanner(radar, scan)
scan_sched.t = np.arange(0, end_t, scan.dwell())

scheduler = ObservedScanning(
    radar = radar,
    controllers = [scan_sched],
)


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


class Scanning(Simulation):
    def __init__(self, population, *args, **kwargs):
        self.population = population
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
            start_t = self.scheduler.controllers[0].t.min(),
            end_t = self.scheduler.controllers[0].t.max(),
            max_dpos=1e3,
        )
        state = obj.get_state(t)
        return state, t

    @MPI_action(action='barrier')
    @iterable_cache(steps='propagate', caches='h5', MPI=True, log=True, reduce=lambda t,x: None)
    @cached_step(caches='pickle')
    def find_passes(self, index, item):
        state, t = item
        passes = scheduler.radar.find_passes(t, state, cache_data = False)
        return passes

    @MPI_action(action='barrier')
    @iterable_cache(steps='passes', caches='pickle', MPI=True, log=True, reduce=lambda t,x: None)
    @cached_step(caches='pickle')
    def observe_passes(self, index, item):
        data = scheduler.observe_passes(item, space_object = self.population.get_object(index), snr_limit=False)
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





sim = Scanning(
    population = pop,
    scheduler = scheduler,
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
sim.logger.always('\n'+sim.profiler.fmt(normalize='total'))

print(f'Total detected: {sim.total_detected()}')

sim.plot()

plt.show()


## EX

# sim.run('propagate')
# sim.run('passes')
# for ind, freq in enumerate([1.2e6, 2.4e6]):
#     sim.checkout('master')
#     sim.branch(f'f{ind}')
#     sim.scheduler.radar.tx[0].beam.frequency = freq
#     sim.scheduler.radar.rx[0].beam.frequency = freq
#     sim.run('observe')

##


