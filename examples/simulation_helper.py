#!/usr/bin/env python

'''
Simulate scanning for objects
======================================
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pickle

import sorts
eiscat3d = sorts.radars.eiscat3d_interp

from sorts.scheduler import StaticList, ObservedParameters
from sorts.controller import Scanner
from sorts import SpaceObject, Simulation, simulation_step
from sorts.radar.scans import Fence

from sorts.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)

end_t = 600.0
scan = Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)

objs = [
    SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = a, 
        e = 0.1, 
        i = 75, 
        raan = 79,
        aop = 0,
        mu0 = 60,
        mjd0 = 53005.0,
        d = 0.3,
    ) for a in np.linspace(7200e3, 8400e3, 4)
]

class ObservedScanning(StaticList, ObservedParameters):
    pass

scan_sched = Scanner(eiscat3d, scan)
scan_sched.t = np.arange(0, end_t, scan.dwell())

scheduler = ObservedScanning(
    radar = eiscat3d, 
    controllers = [scan_sched], 
)

class Scanning(Simulation):
    def __init__(self, objs, *args, **kwargs):
        self.objs = objs

        super().__init__(*args, **kwargs)

        self.steps['propagate'] = self.get_states
        self.steps['passes'] = self.find_passes
        self.steps['observe'] = self.observe_passes


    @simulation_step(iterable='objs', store='props', MPI=True, h5_cache=True)
    def get_states(self, index, item):
        t = sorts.equidistant_sampling(
            orbit = item.orbit, 
            start_t = self.scheduler.controllers[0].t.min(), 
            end_t = self.scheduler.controllers[0].t.max(), 
            max_dpos=1e3,
        )
        state = item.get_state(t)
        return {'t': t, 'state': state}

    @simulation_step(iterable='props', store='passes', custom_cache='passes', MPI=True)
    def find_passes(self, index, item):
        passes = scheduler.radar.find_passes(item['t'], item['state'], cache_data = False)
        return {'passes': passes}

    @simulation_step(iterable='passes', store='obs_data', MPI=True, pickle_cache=True, MPI_mode='gather')
    def observe_passes(self, index, item):
        data = scheduler.observe_passes(item['passes'], space_object = self.objs[index], snr_limit=True)
        return {'data': data}


    def save_passes(self, path, passes):
        with open(path, 'wb') as h:
            pickle.dump(passes, h)

    def load_passes(self, path):
        with open(path, 'rb') as h:
            passes = pickle.load(h)
        return passes



sim = Scanning(
    objs = objs,
    scheduler = scheduler,
    root = '/home/danielk/IRF/E3D_PA/sorts_v4_tests/sim1',
)
# sim.delete('master')

sim.profiler.start('total')

sim.run()

sim.profiler.stop('total')
sim.logger.always('\n'+sim.profiler.fmt(normalize='total'))



