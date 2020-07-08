#!/usr/bin/env python

'''

'''

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate

import sorts
from sorts.propagator import Orekit
from sorts.radar.instances import eiscat3d
from sorts.scheduler import PriorityTracking, ObservedParameters
from sorts import SpaceObject
from sorts.profiling import Profiler


orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'
opts = dict(
    orekit_data = orekit_data, 
    settings=dict(
        in_frame='EME',
        out_frame='ITRF',
        drag_force = False,
        radiation_pressure = False,
    ),
)


logger = sorts.profiling.get_logger('tracking')

objs = [
    SpaceObject(
        Orekit,
        propagator_options = opts,
        a = 7200e3, 
        e = 0.1, 
        i = 75, 
        raan = 79,
        aop = 0,
        mu0 = 60,
        mjd0 = 53005.0,
        d = 1.0,
    ),
    SpaceObject(
        Orekit,
        propagator_options = opts,
        a = 7200e3, 
        e = 0.1, 
        i = 69, 
        raan = 74,
        aop = 0,
        mu0 = 0,
        mjd0 = 53005.0,
        d = 1.0,
    )
]

for obj in objs: print(obj)

class ObservedTracking(PriorityTracking, ObservedParameters):
    def generate_schedule(self, t, generator):
        data = np.empty((len(t),len(self.radar.rx)*2+1), dtype=np.float64)
        data[:,0] = t
        for ind,radar in enumerate(generator):
            for ri, rx in enumerate(radar.rx):
                data[ind,1+ri*2] = rx.beam.azimuth
                data[ind,2+ri*2] = rx.beam.elevation
        return data


scheduler = ObservedTracking(
    radar = eiscat3d, 
    space_objects = objs, 
    timeslice = 0.1, 
    allocation = 0.1*200, 
    end_time = 3600*6.0,
    priority = [0.2, 1.0],
    logger = logger,
)


sched_data = scheduler.schedule()
rx_head = [f'rx{i} {co}' for i in range(len(scheduler.radar.rx)) for co in ['az', 'el']]
sched_tab = tabulate(sched_data, headers=["t [s]"] + rx_head)

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