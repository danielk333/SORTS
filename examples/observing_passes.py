#!/usr/bin/env python

'''

'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt

import sorts
eiscat3d = sorts.radars.eiscat3d
from sorts.controller import Tracker
from sorts.scheduler import StaticList, ObservedParameters
from sorts import SpaceObject
from sorts.profiling import Profiler

from sorts.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)
prop = Prop_cls(**Prop_opts)


objs = [
    SpaceObject(
        Prop_cls,
        propagator_options = Prop_opts,
        a = 7200e3, 
        e = 0.1, 
        i = 75, 
        raan = 79,
        aop = 0,
        mu0 = mu0,
        mjd0 = 53005.0,
        d = 1.0,
    )
    for mu0 in [62.0, 61.9]
]

for obj in objs: print(obj)

t = sorts.equidistant_sampling(
    orbit = objs[0].orbit, 
    start_t = 0, 
    end_t = 3600*6, 
    max_dpos=1e3,
)

print(f'Temporal points: {len(t)}')
states0 = objs[0].get_state(t)
states1 = objs[1].get_state(t)

#set cache_data = True to save the data in local coordinates 
#for each pass inside the Pass instance, setting to false saves RAM
passes0 = eiscat3d.find_passes(t, states0, cache_data = False) 
passes1 = eiscat3d.find_passes(t, states1, cache_data = False)


#just create a controller for observing 10 points of the first pass
ps = passes0[0][0][0]
use_inds = np.arange(0,len(ps.inds),len(ps.inds)//10)
e3d_tracker = Tracker(radar = eiscat3d, t=t[ps.inds[use_inds]], ecefs=states0[:3,ps.inds[use_inds]])

class MyStaticList(StaticList, ObservedParameters):

    def __init__(self, radar, controllers, profiler=None, logger=None):
        super().__init__(
            radar=radar, 
            controllers=controllers, 
            profiler=profiler,
            logger=logger,
        )

    def generate_schedule(self, t, generator):
        data = np.empty((len(t),len(self.radar.rx)*2+2), dtype=np.float64)
        data[:,0] = t
        data[:,len(self.radar.rx)*2+1] = 0.2
        for ind,radar in enumerate(generator):
            for ri, rx in enumerate(radar.rx):
                data[ind,1+ri*2] = rx.beam.azimuth
                data[ind,2+ri*2] = rx.beam.elevation
        return data


p = Profiler()

scheduler = MyStaticList(radar = eiscat3d, controllers=[e3d_tracker], profiler=p)

sched_data = scheduler.schedule()
from tabulate import tabulate

rx_head = [f'rx{i} {co}' for i in range(len(scheduler.radar.rx)) for co in ['az', 'el']]
sched_tab = tabulate(sched_data, headers=["t [s]"] + rx_head + ['dwell'])

print(sched_tab)

p.start('total')
data0 = scheduler.observe_passes(passes0, space_object = objs[0], snr_limit=False)
p.stop('total')
print(p.fmt(normalize='total'))

data1 = scheduler.observe_passes(passes1, space_object = objs[1], snr_limit=False)

#create a tdm file 


pth = pathlib.Path(__file__).parent / 'data' / 'test_tdm.tdm'
print(f'Writing TDM data to: {pth}')

dat = data0[0][0][0]
sorts.io.write_tdm(
    pth,
    dat['t'],
    dat['range'],
    dat['range_rate'],
    np.ones(dat['range'].shape),
    np.ones(dat['range_rate'].shape),
    freq=eiscat3d.tx[0].beam.frequency,
    tx_ecef=eiscat3d.tx[0].ecef,
    rx_ecef=eiscat3d.rx[0].ecef,
    tx_name="EISCAT 3D Skiboten",
    rx_name="EISCAT 3D Skiboten",
    oid="Some cool space object",
    tdm_type="track",
)


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

for tx in scheduler.radar.tx:
    axes[0][0].plot([tx.ecef[0]],[tx.ecef[1]],[tx.ecef[2]], 'or')
for rx in scheduler.radar.rx:
    axes[0][0].plot([rx.ecef[0]],[rx.ecef[1]],[rx.ecef[2]], 'og')

for pi in range(len(passes0[0][0])):
    dat = data0[0][0][pi]
    dat2 = data1[0][0][pi]
    if dat is not None:
        axes[0][0].plot(states0[0,passes0[0][0][pi].inds], states0[1,passes0[0][0][pi].inds], states0[2,passes0[0][0][pi].inds], '-', label=f'pass-{pi}')
        axes[0][1].plot(dat['t']/3600.0, dat['range'], '-', label=f'pass-{pi}')
        axes[1][0].plot(dat['t']/3600.0, dat['range_rate'], '-', label=f'pass-{pi}')
        axes[1][1].plot(dat['t']/3600.0, 10*np.log10(dat['snr']), '-', label=f'pass-{pi}')
    if dat2 is not None:
        axes[0][0].plot(states1[0,passes1[0][0][pi].inds], states1[1,passes1[0][0][pi].inds], states1[2,passes1[0][0][pi].inds], '-', label=f'obj2 pass-{pi}')
        axes[0][1].plot(dat2['t']/3600.0, dat2['range'], '-', label=f'obj2 pass-{pi}')
        axes[1][0].plot(dat2['t']/3600.0, dat2['range_rate'], '-', label=f'obj2 pass-{pi}')
        axes[1][1].plot(dat2['t']/3600.0, 10*np.log10(dat2['snr']), '-', label=f'obj2 pass-{pi}')

axes[0][1].legend()
plt.show()