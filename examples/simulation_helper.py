#!/usr/bin/env python

'''
Simulate scanning for objects
======================================
'''

import numpy as np
import matplotlib.pyplot as plt
import h5py

import sorts
eiscat3d = sorts.radars.eiscat3d_interp

from sorts.scheduler import StaticList, ObservedParameters
from sorts.controller import Scanner
from sorts import SpaceObject, Simulation
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
        a = 7200e3, 
        e = 0.1, 
        i = 75, 
        raan = 79,
        aop = 0,
        mu0 = 60,
        mjd0 = 53005.0,
        d = 0.3,
    )
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
        if 'paths' in kwargs:
            if 'objs' not in kwargs['paths']:
                kwargs['paths'].append('objs')
        else:
            kwargs['paths'] = ['logs', 'objs']
        super().__init__(*args, **kwargs)


        self.steps['get_states'] = self.get_states
        self.steps['find_passes'] = self.find_passes
        self.steps['observe_passes'] = self.observe_passes

        self.datas = [None]*len(objs)
        self.passes = [None]*len(objs)
        self.states = [None]*len(objs)
        self.ts = [None]*len(objs)

    def get_states(self):

        for ind in range(len(self.objs)):
            fname = self.get_path('objs') / f'obj{ind}.h5'

            if fname.is_file():
                with h5py.File(fname,'r') as h:
                    t = h['t'][()]
                    state = h['state'][()]
            else:
                t = sorts.equidistant_sampling(
                    orbit = self.objs[ind].orbit, 
                    start_t = self.scheduler.controllers[0].t.min(), 
                    end_t = self.scheduler.controllers[0].t.max(), 
                    max_dpos=1e3,
                )
                state = self.objs[ind].get_state(t)
                with h5py.File(fname,'w') as h:
                    h.create_dataset('t', data=t)
                    h.create_dataset('state', data=state)

            self.states[ind] = state
            self.ts[ind] = t

    def find_passes(self):
        for ind in range(len(self.objs)):
            self.passes[ind] = scheduler.radar.find_passes(self.ts[ind], self.states[ind], cache_data = False)

    def observe_passes(self):
        for ind in range(len(self.objs)):
            self.datas[ind] = scheduler.observe_passes(self.passes[ind], space_object = self.objs[ind], snr_limit=True)



sim = Scanning(
    objs = objs,
    scheduler = scheduler,
    root = '/home/danielk/IRF/E3D_PA/sorts_v4_tests/sim1',
)
sim.profiler.start('total')

sim.run()

sim.profiler.stop('total')
sim.logger.always(sim.profiler.fmt(normalize='total'))



