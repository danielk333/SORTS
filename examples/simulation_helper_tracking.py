#!/usr/bin/env python

'''
Simulate tracking with simulation helper
==========================================
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
        d = 1.0,
        oid = 1,
    ),
    SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = 7200e3, 
        e = 0.1, 
        i = 69, 
        raan = 74,
        aop = 0,
        mu0 = 0,
        mjd0 = 53005.0,
        d = 1.0,
        oid = 2,
    )
]

for obj in objs: print(obj)

class ObservedTracking(PriorityTracking, ObservedParameters):
    pass


scheduler = ObservedTracking(
    radar = eiscat3d, 
    space_objects = objs, 
    timeslice = 0.1, 
    allocation = 0.1*200, 
    end_time = 3600*6.0,
    priority = [0.2, 1.0],
)

class Tracking(Simulation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.obj_inds = range(len(self.scheduler.space_objects))

        self.steps['find_passes'] = self.find_passes
        self.steps['find_passes'] = self.find_passes

    @simulation_step(iterable='obj_inds', store=None, MPI=True, caches=['pickle'])
    def find_passes(self, index, item):
        self.scheduler.get_passes(item)

    @MPI_mono_simulation_step(process_id = 0)
    @simulation_step(caches=['pickle'])
    def get_measurnments(self):

        data = Simulation.bcast(process_id = 0, data)


sim = Tracking(
    scheduler = scheduler,
    root = '/home/danielk/IRF/E3D_PA/sorts_v4_tests/sim2',
)
sim.delete('test')
sim.branch('test', empty=True)

sim.profiler.start('total')

sim.run()

sim.profiler.stop('total')
sim.logger.always('\n'+sim.profiler.fmt(normalize='total'))



