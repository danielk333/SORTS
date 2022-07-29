#!/usr/bin/env python

'''
==========================================
Static radar controller and time sampling
==========================================

You can speed up execution by switching radar controller that is optimized for the use case.
Also reducing time sampling of the radar controller can speed up calculations.

'''

import numpy as np
import matplotlib.pyplot as plt


import sorts
eiscat3d = sorts.radars.eiscat3d_interp

from sorts.radar.scheduler import StaticList, ObservedParameters
from sorts.radar.controllers import Static, Scanner
from sorts import SpaceObject
from sorts.common.profiling import Profiler
from sorts.radar.scans import Beampark

from sorts.targets.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)

end_t = 600.0

logger = sorts.profiling.get_logger('scanning')

obj = SpaceObject(
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
)

print(obj)
class ObservedScanning(StaticList, ObservedParameters):
    pass

static_ctrl = Static(eiscat3d, azimuth=0, elevation=90, logger=logger, meta={'dwell': 0.1})
static_ctrl.t = np.arange(0, end_t, static_ctrl.meta['dwell'])

static_ctrl_undersamp = Static(eiscat3d, azimuth=0, elevation=90, logger=logger, meta={'dwell': 0.1})
static_ctrl_undersamp.t = np.arange(0, end_t, 1.0)

scan = Beampark(azimuth=0, elevation=90, dwell=0.1)
scanner_ctrl = Scanner(eiscat3d.copy(), scan, logger=logger)
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

    p.start('equidistant_sampling')
    t = sorts.equidistant_sampling(
        orbit = obj.state, 
        start_t = 0, 
        end_t = end_t, 
        max_dpos=1e3,
    )
    p.stop('equidistant_sampling')

    print(f'Temporal points obj: {len(t)}')
    
    p.start('get_state')
    states = obj.get_state(t)
    p.stop('get_state')

    p.start('find_passes')
    #rename cache_data to something more descriptive
    passes = eiscat3d.find_passes(t, states, cache_data = True)
    p.stop('find_passes')

    p.start('observe_passes')
    data = scheduler.observe_passes(passes, space_object = obj, snr_limit=False)
    p.stop('observe_passes')

    for psi in data:
        for txps in psi:
            for rxtxps in txps:
                print(f'Max SNR={10*np.log10(rxtxps["snr"].max())} dB')

    p.stop('total')
    print(f'\n {radar_ctrl.__class__}: len(t) = {len(radar_ctrl.t)} \n')
    print(p.fmt(normalize='total'))

run_scanning_simulation(scanner_ctrl)
run_scanning_simulation(static_ctrl)
run_scanning_simulation(static_ctrl_undersamp)