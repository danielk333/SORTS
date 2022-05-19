#!/usr/bin/env python

'''
An example scheduler for tracking
======================================
'''

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from astropy.time import Time

import sorts

from sorts.radar.scheduler import PriorityTracking, ObservedParameters
from sorts import SpaceObject
from sorts.common.profiling import Profiler
from sorts.targets.propagator import SGP4


eiscat3d = sorts.radars.eiscat3d


poptions = dict(
    settings = dict(
        in_frame='GCRS',
        out_frame='ITRS',
    ),
)

epoch = Time(53005.0, format='mjd')

logger = sorts.profiling.get_logger('tracking')

objs = [
    SpaceObject(
        SGP4,
        propagator_options = poptions,
        a = 7200e3, 
        e = 0.01, 
        i = 75, 
        raan = 79,
        aop = 0,
        mu0 = 60,
        epoch = epoch,
        parameters = dict(
            d = 1.0,
        ),
        oid = 1,
    ),
    SpaceObject(
        SGP4,
        propagator_options = poptions,
        a = 7200e3, 
        e = 0.01, 
        i = 69, 
        raan = 74,
        aop = 0,
        mu0 = 0,
        epoch = epoch,
        parameters = dict(
            d = 1.0,
        ),
        oid = 42,
    )
]

for obj in objs: print(obj)

class ObservedTracking(PriorityTracking, ObservedParameters):
    def generate_schedule(self, t, generator):
        data = np.empty((len(t),len(self.radar.rx)*2+1), dtype=np.float64)
        data[:,0] = t
        names = []
        targets = []
        for ind,mrad in enumerate(generator):
            radar, meta = mrad
            names.append(meta['controller_type'].__name__)
            targets.append(meta['target'])
            for ri, rx in enumerate(radar.rx):
                data[ind,1+ri*2] = rx.beam.azimuth
                data[ind,2+ri*2] = rx.beam.elevation
        data = data.T.tolist() + [names, targets]
        data = list(map(list, zip(*data)))
        return data


scheduler = ObservedTracking(
    radar = eiscat3d, 
    space_objects = objs, 
    timeslice = 0.1, 
    allocation = 0.1*50, 
    end_time = 3600*12.0,
    max_dpos = 1e4,
    epoch = epoch,
    priority = [0.2, 1.0],
    logger = logger,
    use_pass_states = True,
    collect_passes = True,
)

scheduler.update()
scheduler.set_measurements()


sched_data = scheduler.schedule(start=3600*2.0, stop=3600*9.0)
#Without start and stop argument entire schedule gets printed
#sched_data = scheduler.schedule()
rx_head = [f'rx{i} {co}' for i in range(len(scheduler.radar.rx)) for co in ['az', 'el']]
sched_tab = tabulate(sched_data, headers=["t [s]"] + rx_head + ['Controller', 'Target'])

print(sched_tab)

datas = []
for ind in range(len(objs)):
    data = scheduler.observe_passes(scheduler.passes[ind], space_object = objs[ind], snr_limit=False)
    datas.append(data)


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

for ind in range(len(objs)):
    for pi, ps in enumerate(scheduler.passes[ind][0][0]):

        zang = ps.zenith_angle()
        snr = ps.calculate_snr(eiscat3d.tx[0], eiscat3d.rx[0], diameter=objs[ind].d)

        axes[0][0].plot(ps.enu[0][0,:], ps.enu[0][1,:], ps.enu[0][2,:], '-', label=f'pass-{pi}')
        axes[0][0].set_xlabel('East-North-Up coordinates')

        axes[0][1].plot((ps.t - ps.start())/3600.0, zang[0], '-', label=f'pass-{pi}')
        axes[0][1].set_xlabel('Time past epoch [h]')
        axes[0][1].set_ylabel('Zenith angle from TX [deg]')

        axes[1][0].plot((ps.t - ps.start())/3600.0, ps.range()[0]*1e-3, '-', label=f'pass-{pi}')
        axes[1][0].set_xlabel('Time past epoch [h]')
        axes[1][0].set_ylabel('Range from TX [km]')

        axes[1][1].plot((ps.t - ps.start())/3600.0, 10*np.log10(snr), '-', label=f'optimal-pass-{pi}')
        axes[1][1].plot((datas[ind][0][0][pi]['t'] - ps.start())/3600.0, 10*np.log10(datas[ind][0][0][pi]['snr']), '.', label=f'observed-pass-{pi}')
        axes[1][1].set_xlabel('Time past epoch [h]')
        axes[1][1].set_ylabel('SNR [dB]')

axes[1][1].legend()
plt.show()
