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
from sorts.controller import Scanner
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
scan = Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)

p = Profiler()

logger = sorts.profiling.get_logger('scanning')

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
        d = 0.3,
    )
]

objs = objs[:1]

for obj in objs: print(obj)

class ObservedScanning(StaticList, ObservedParameters):
    pass

scan_sched = Scanner(eiscat3d, scan)
scan_sched.t = np.arange(0, end_t, scan.dwell())

p.start('total')
scheduler = ObservedScanning(
    radar = eiscat3d, 
    controllers = [scan_sched], 
    logger = logger,
    profiler = p,
)



datas = []
passes = []
states = []
for ind in range(len(objs)):
    p.start('equidistant_sampling')
    t = sorts.equidistant_sampling(
        orbit = objs[ind].orbit, 
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
    passes += [eiscat3d.find_passes(t, states[ind], cache_data = True)]
    p.stop('find_passes')

    p.start('observe_passes')
    data = scheduler.observe_passes(passes[ind], space_object = objs[ind], snr_limit=True)
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

sorts.plotting.grid_earth(axes[0][0])
for tx in eiscat3d.tx:
    axes[0][0].plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]],'or')
for rx in eiscat3d.rx:
    axes[0][0].plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]],'og')

for radar, meta in scan_sched(np.arange(0,scan.cycle(),scan.dwell())):
    for tx in radar.tx:
        point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*scan_sched.r.max() + tx.ecef
        axes[0][0].plot([tx.ecef[0], point_tx[0]], [tx.ecef[1], point_tx[1]], [tx.ecef[2], point_tx[2]], 'r-', alpha=0.15)

        for rx in radar.rx:
            pecef = rx.pointing_ecef/np.linalg.norm(rx.pointing_ecef, axis=0)

            for ri in range(pecef.shape[1]):
                point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*scan_sched.r[ri] + tx.ecef
                point = pecef[:,ri]*np.linalg.norm(rx.ecef - point_tx) + rx.ecef

                axes[0][0].plot([rx.ecef[0], point[0]], [rx.ecef[1], point[1]], [rx.ecef[2], point[2]], 'g-', alpha=0.05)

for ind in range(len(objs)):
    for pi in range(len(passes[ind][0][0])):
        ps = passes[ind][0][0][pi]
        dat = datas[ind][0][0][pi]
        
        axes[0][0].plot(states[ind][0,ps.inds], states[ind][1,ps.inds], states[ind][2,ps.inds], '-')

        if dat is not None:
            axes[0][1].plot(dat['t']/3600.0, dat['range']*1e-3, '-', label=f'obj{ind}-pass{pi}')
            axes[1][0].plot(dat['t']/3600.0, dat['range_rate']*1e-3, '-')
            axes[1][1].plot(dat['t']/3600.0, 10*np.log10(dat['snr']), '-')


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