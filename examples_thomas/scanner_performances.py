#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 18 14:15:39 2022

@author: thomas
"""

import numpy as np
import matplotlib.pyplot as plt

from sorts.radar.scans import Fence
from sorts.targets.propagator import Kepler
from sorts.targets import SpaceObject
from sorts.radar.system import instances
from sorts.radar import scheduler, controllers
from sorts import equidistant_sampling

from sorts.common import profiling
from sorts import plotting

import pyorb

# Computation / test setup
OBJECT_NUMBER = 1
CONTROLLER_NUMBER = 1

end_t = 600

# Scan type definition
scan = Fence(azimuth=90, min_elevation=30, dwell=0.1, num=40)

# RADAR definition
eiscat3d = instances.eiscat3d

# Object definition
# Propagator
Prop_cls = Kepler
Prop_opts = dict(
    settings = dict(
        out_frame='ITRS',
        in_frame='TEME',
    ),
)

# Object properties
orbits_a = np.array([7200, 8500, 12000, 10000])*1e3 # km
orbits_i = np.array([80, 105, 105, 80]) # deg
orbits_raan = np.array([86, 160, 180, 90]) # deg
orbits_aop = np.array([0, 50, 40, 55]) # deg
orbits_mu0 = np.array([60, 5, 30, 8]) # deg


if (OBJECT_NUMBER > len(orbits_a)): 
    OBJECT_NUMBER = len(orbits_a)
    
objects = []
orb = []
for obj_id in range(OBJECT_NUMBER):
    pyorb.Orbit

    objects.append(SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = orbits_a[obj_id], 
        e = 0.1,
        i = orbits_i[obj_id],
        raan = orbits_raan[obj_id],
        aop = orbits_aop[obj_id],
        mu0 = orbits_mu0[obj_id],
        
        epoch = 53005.0,
        parameters = dict(
            d = 0.1,
        ),
    ))
    
    print(objects[obj_id])
    
# Profiler
p = profiling.Profiler()
logger = profiling.get_logger('scanning')

# Starting simulation
class ObservedScanning(scheduler.StaticList, scheduler.ObservedParameters):
    pass

scanner_ctrl = controllers.Scanner(eiscat3d, scan, t = np.arange(0, end_t, scan.dwell()), profiler=p, logger=logger)

p.start('total')
scheduler = ObservedScanning(
    radar = eiscat3d, 
    controllers = [scanner_ctrl for ctrl_id in range(CONTROLLER_NUMBER)], 
    logger = logger,
    profiler = p,
)

datas = []
passes = []
states = []

for obj_id in range(len(objects)):
    # Create equidistant points for state sampling
    p.start('equidistant_sampling')
    t = equidistant_sampling(
        orbit = objects[obj_id].state, 
        start_t = 0, 
        end_t = end_t, 
        max_dpos=50e3,
    )
    p.stop('equidistant_sampling')

    print(f'Temporal points obj {obj_id}: {len(t)}')
    
    p.start('get_state')
    states.append(objects[obj_id].get_state(t))
    p.stop('get_state')
    
    # look for passes in the state vectors of the objects
    p.start('find_passes')
    passes += [eiscat3d.find_passes(t, states[obj_id], cache_data=True)] 
    p.stop('find_passes')

    # gather measurements from passes
    p.start('observe_passes')
    data = scheduler.observe_passes(passes[obj_id], space_object=objects[obj_id], snr_limit=False)
    p.stop('observe_passes')
    
    datas.append(data)

p.stop('total')
print(p.fmt(normalize='total'))

fig = plt.figure(figsize=(15,15))
axes = [
    [
        fig.add_subplot(221, projection='3d'),
        fig.add_subplot(222),
    ],
    [
        fig.add_subplot(223),
        fig.add_subplot(224),
    ],
]

plotting.grid_earth(axes[0][0])

for tx in eiscat3d.tx:
    axes[0][0].plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in eiscat3d.rx:
    axes[0][0].plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

for radar, meta in scanner_ctrl(np.arange(0,scan.cycle(),scan.dwell())):
    for tx in radar.tx:
        point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*scanner_ctrl.r.max() + tx.ecef
        axes[0][0].plot([tx.ecef[0], point_tx[0]], [tx.ecef[1], point_tx[1]], [tx.ecef[2], point_tx[2]], 'r-', alpha=0.15)

        for rx in radar.rx:
            pecef = rx.pointing_ecef/np.linalg.norm(rx.pointing_ecef, axis=0)
            
            if len(np.shape(pecef)) <= 1: 
                pecef.shape = (3,1)

            for ri in range(pecef.shape[1]):
                point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*scanner_ctrl.r[ri] + tx.ecef
                point = pecef[:,ri]*np.linalg.norm(rx.ecef - point_tx) + rx.ecef

                axes[0][0].plot([rx.ecef[0], point[0]], [rx.ecef[1], point[1]], [rx.ecef[2], point[2]], 'g-', alpha=0.05)

for obj_id in range(len(objects)):
    for pi in range(len(passes[obj_id][0][0])):
        ps = passes[obj_id][0][0][pi]
        dat = datas[obj_id][0][0][pi]
        
        axes[0][0].plot(states[obj_id][0,:], states[obj_id][1,:], states[obj_id][2,:], '-b')
        axes[0][0].plot(states[obj_id][0,ps.inds], states[obj_id][1,ps.inds], states[obj_id][2,ps.inds], '-r')

        if dat is not None:

            SNRdB = 10*np.log10(dat['snr'])
            det_inds = SNRdB > 10.0

            axes[0][1].plot(dat['t']/3600.0, dat['range']*1e-3, '-', label=f'obj{obj_id}-pass{pi}')
            axes[1][0].plot(dat['t']/3600.0, dat['range_rate']*1e-3, '-')
            axes[1][1].plot(dat['t']/3600.0, SNRdB, '-')

            axes[0][1].plot(dat['t'][det_inds]/3600.0, dat['range'][det_inds]*1e-3, '.r')
            axes[1][0].plot(dat['t'][det_inds]/3600.0, dat['range_rate'][det_inds]*1e-3, '.r')
            axes[1][1].plot(dat['t'][det_inds]/3600.0, SNRdB[det_inds], '.r')
            axes[1][1].set_ylim([0, None])


font_ = 18
axes[0][1].set_xlabel('Time [h]', fontsize=font_)
axes[1][0].set_xlabel('Time [h]', fontsize=font_)
axes[1][1].set_xlabel('Time [h]', fontsize=font_)

axes[0][1].set_ylabel('Two way range [km]', fontsize=font_)
axes[1][0].set_ylabel('Two way range rate [km/s]', fontsize=font_)
axes[1][1].set_ylabel('SNR [dB]', fontsize=font_)

axes[0][1].legend()

dr = 600e3
axes[0][0].set_xlim([eiscat3d.tx[0].ecef[0]-dr, eiscat3d.tx[0].ecef[0]+dr])
axes[0][0].set_ylim([eiscat3d.tx[0].ecef[1]-dr, eiscat3d.tx[0].ecef[1]+dr])
axes[0][0].set_zlim([eiscat3d.tx[0].ecef[2]-dr, eiscat3d.tx[0].ecef[2]+dr])

plt.show()