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
eiscat3d = sorts.radars.eiscat3d
from sorts.scheduler import PriorityTracking, ObservedParameters
from sorts import SpaceObject
from sorts.profiling import Profiler


from sorts.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)

logger = sorts.profiling.get_logger('tracking')

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
    allocation = 0.1*200, 
    end_time = 3600*6.0,
    epoch = Time(53005.0, format='mjd'),
    priority = [0.2, 1.0],
    logger = logger,
)

scheduler.update()
scheduler.set_measurements()


sched_data = scheduler.schedule()
rx_head = [f'rx{i} {co}' for i in range(len(scheduler.radar.rx)) for co in ['az', 'el']]
sched_tab = tabulate(sched_data, headers=["t [s]"] + rx_head + ['Controller', 'Target'])

print(sched_tab)

datas = []
for ind in range(len(objs)):
    data = scheduler.observe_passes(scheduler.passes[ind], space_object = objs[ind], snr_limit=False)
    datas.append(data)

# print(data)

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
    for pi in range(len(scheduler.passes[ind][0][0])):
        ps = scheduler.passes[ind][0][0][pi]
        axes[0][0].plot(ps.enu[0][0,:], ps.enu[0][1,:], ps.enu[0][2,:], '-')
        dat = datas[ind][0][0][pi]

        if dat is not None:
            axes[0][1].plot(dat['t']/3600.0, dat['range'], '-', label=f'obj{ind}-pass{pi}')
            axes[1][0].plot(dat['t']/3600.0, dat['range_rate'], '-')
            axes[1][1].plot(dat['t']/3600.0, 10*np.log10(dat['snr']), '-')


axes[0][1].legend()
plt.show()