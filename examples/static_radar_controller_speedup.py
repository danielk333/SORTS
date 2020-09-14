#!/usr/bin/env python

'''
Simulate scanning for objects
======================================
'''

import numpy as np
import matplotlib.pyplot as plt


import sorts
eiscat3d = sorts.radars.eiscat3d_interp

from sorts.scheduler import StaticList, ObservedParameters
from sorts.controller import Static, Scanner
from sorts import SpaceObject
from sorts.profiling import Profiler
from sorts.radar.scans import Fence

from sorts.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)

end_t = 600.0

logger = sorts.profiling.get_logger('scanning')

objs = [
    SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = 7200e3, 
        e = 0.02, 
        i = 75, 
        raan = 86,
        aop = 0,
        mu0 = 60,
        epoch = 53005.0,
        parameters = dict(
            d = 0.1,
        ),
    ),
]


for obj in objs: print(obj)

class ObservedScanning(StaticList, ObservedParameters):
    pass

static_ctrl = Static(eiscat3d, logger=logger)
static_ctrl.t = np.arange(0, end_t, static_ctrl.meta['dwell'])

scan = Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)
scanner_ctrl = Scanner(eiscat3d, scan, logger=logger)
scanner_ctrl.t = np.arange(0, end_t, scan.dwell())

def run_scanning_simulation(radar_ctrl):

    p = Profiler()
    radar_ctrl.profiler = p

    p.start('total')
    scheduler = ObservedScanning(
        radar = eiscat3d, 
        controllers = [radar_ctrl], 
        logger = logger,
        profiler = p,
    )



    datas = []
    passes = []
    states = []
    for ind in range(len(objs)):
        p.start('equidistant_sampling')
        t = sorts.equidistant_sampling(
            orbit = objs[ind].state, 
            start_t = 0, 
            end_t = end_t, 
            max_dpos=1e3,
        )
        p.stop('equidistant_sampling')

        print(f'Temporal points obj {ind}: {len(t)}')
        
        p.start('get_state')
        states += [objs[ind].get_state(t)]
        p.stop('get_state')

        p.start('find_passes')
        #rename cache_data to something more descriptive
        passes += [eiscat3d.find_passes(t, states[ind], cache_data = True)] 
        p.stop('find_passes')

        p.start('observe_passes')
        data = scheduler.observe_passes(passes[ind], space_object = objs[ind], snr_limit=False)
        p.stop('observe_passes')

        datas.append(data)

    p.stop('total')
    print(f'{radar_ctrl.__class__}')
    print(p.fmt(normalize='total'))

run_scanning_simulation(scanner_ctrl)
run_scanning_simulation(static_ctrl)