#!/usr/bin/env python

'''
Optimizing with interpolation
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
from sorts.interpolation import Legendre8

from sorts.propagator import Orekit

orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'

Prop_cls = Orekit
Prop_opts = dict(
    orekit_data = orekit_data, 
    settings = dict(
        in_frame='GCRS',
        out_frame='ITRS',
    ),
)

end_t = 12*3600.0
scan = Fence(azimuth=90, num=40, dwell=0.1, min_elevation=30)

p = Profiler()

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
            A = 1.0,
        ),
    ),
]


for obj in objs: print(obj)

class ObservedScanning(StaticList, ObservedParameters):
    pass

scanner_ctrl = Scanner(eiscat3d, scan, profiler=p, logger=logger)
scanner_ctrl.t = np.arange(0, end_t, scan.dwell())

p.start('total')
scheduler = ObservedScanning(
    radar = eiscat3d, 
    controllers = [scanner_ctrl], 
    logger = logger,
    profiler = p,
)


t = np.arange(0.0,end_t,30.0)

datas = []
passes = []
states = []
for ind in range(len(objs)):

    print(f'Temporal points obj {ind}: {len(t)}')
    
    p.start('get_state')
    states += [objs[ind].get_state(t)]
    p.stop('get_state')

    interpolator = Legendre8(states[ind], t)

    p.start('find_passes')
    #rename cache_data to something more descriptive
    passes += [eiscat3d.find_passes(t, states[ind], cache_data = False)] 
    p.stop('find_passes')

    p.start('observe_passes')
    data = scheduler.observe_passes(passes[ind], space_object = objs[ind], interpolator = None, snr_limit=False)
    p.stop('observe_passes')

    p.start('observe_passes_interpolator')
    data = scheduler.observe_passes(passes[ind], space_object = objs[ind], interpolator = interpolator, snr_limit=False)
    p.stop('observe_passes_interpolator')

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

for radar, meta in scanner_ctrl(np.arange(0,scan.cycle(),scan.dwell())):
    for tx in radar.tx:
        point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*scanner_ctrl.r.max() + tx.ecef
        axes[0][0].plot([tx.ecef[0], point_tx[0]], [tx.ecef[1], point_tx[1]], [tx.ecef[2], point_tx[2]], 'r-', alpha=0.15)

        for rx in radar.rx:
            pecef = rx.pointing_ecef/np.linalg.norm(rx.pointing_ecef, axis=0)

            for ri in range(pecef.shape[1]):
                point_tx = tx.pointing_ecef/np.linalg.norm(tx.pointing_ecef, axis=0)*scanner_ctrl.r[ri] + tx.ecef
                point = pecef[:,ri]*np.linalg.norm(rx.ecef - point_tx) + rx.ecef

                axes[0][0].plot([rx.ecef[0], point[0]], [rx.ecef[1], point[1]], [rx.ecef[2], point[2]], 'g-', alpha=0.05)

for ind in range(len(objs)):
    for pi in range(len(passes[ind][0][0])):
        ps = passes[ind][0][0][pi]
        dat = datas[ind][0][0][pi]
        
        axes[0][0].plot(states[ind][0,ps.inds], states[ind][1,ps.inds], states[ind][2,ps.inds], '-')

        if dat is not None:

            SNRdB = 10*np.log10(dat['snr'])
            det_inds = SNRdB > 10.0

            axes[0][1].plot(dat['t']/3600.0, dat['range']*1e-3, '-', label=f'obj{ind}-pass{pi}')
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