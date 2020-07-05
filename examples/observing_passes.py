#!/usr/bin/env python

'''

'''

import numpy as np
import matplotlib.pyplot as plt
import pyorb

import sorts
from sorts.propagator import Orekit
from sorts.radar.instances import eiscat3d
from sorts.controller import Tracker
from sorts.scheduler import StaticList
from sorts import Pass


orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees=True, a=7200e3, e=0.1, i=75, omega=0, Omega=79, anom=72, epoch=53005.0)
print(orb)

t = sorts.equidistant_sampling(
    orbit = orb, 
    start_t = 0, 
    end_t = 3600*24*1, 
    max_dpos=1e3,
)

orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'
prop = Orekit(
    orekit_data = orekit_data, 
    settings=dict(
        in_frame='EME',
        out_frame='ITRF',
        drag_force = False,
        radiation_pressure = False,
        max_step=10.0,
    ),
)
print(f'Temporal points: {len(t)}')
states = prop.propagate(t, orb.cartesian[:,0], orb.epoch, A=1.0, C_R = 1.0, C_D = 1.0)

passes = eiscat3d.find_passes(t, states)


#just the first pass
ps = passes[0][0][0]
use_inds = np.arange(0,len(ps.t),len(ps.t)//10)
e3d_tracker = Tracker(radar = eiscat3d, t=ps.t[use_inds], ecefs=states[:3,ps.inds[use_inds]])

class MyStaticList(StaticList):

    def __init__(self, radar, controllers, propagator):
        super().__init__(radar, controllers)
        self.propagator = propagator

    def generate_schedule(self, t, generator):
        data = np.empty((len(t),len(self.radar.rx)*2+2), dtype=np.float64)
        data[:,0] = t
        data[:,len(self.radar.rx)*2+1] = 0.2
        for ind,radar in enumerate(generator):
            for ri, rx in enumerate(radar.rx):
                data[ind,1+ri*2] = rx.beam.azimuth
                data[ind,2+ri*2] = rx.beam.elevation
        return data

    def calculate_observation(self, txrx_pass, t, generator, **kwargs):
        orb = kwargs['orbit']
        states = prop.propagate(t, orb.cartesian[:,0], orb.epoch, A=1.0, C_R = 1.0, C_D = 1.0)

        snr = np.empty((len(t),), dtype=np.float64)
        txi, rxi = txrx_pass.station_id

        enus = [
            self.radar.tx[txi].enu(states),
            self.radar.rx[rxi].enu(states),
        ]
        ranges = [Pass.calculate_range(enu) for enu in enus]
        range_rates = [Pass.calculate_range_rate(enu) for enu in enus]
        zang = Pass.calculate_zenith_angle(enus[0])

        for ti, radar in enumerate(generator):
            if radar.tx[txi].enabled and radar.rx[rxi].enabled:

                snr[ti] = sorts.hard_target_snr(
                    radar.tx[txi].beam.gain(enus[0][:3,ti]),
                    radar.rx[rxi].beam.gain(enus[1][:3,ti]),
                    radar.rx[rxi].wavelength,
                    radar.tx[txi].power,
                    ranges[0][ti],
                    ranges[1][ti],
                    diameter_m=kwargs['diameter'],
                    bandwidth=radar.tx[txi].coh_int_bandwidth,
                    rx_noise_temp=radar.rx[rxi].noise,
                )

            else:
                snr[ti] = np.nan

        data = dict(
            t = t,
            snr = snr,
            range = ranges[0] + ranges[1],
            range_rate = range_rates[0] + range_rates[1],
            tx_zenith = zang,
        )
        return data


scheduler = MyStaticList(radar = eiscat3d, controllers=[e3d_tracker], propagator = prop)

sched_data = scheduler.schedule()
from tabulate import tabulate

rx_head = [f'rx{i} {co}' for i in range(len(scheduler.radar.rx)) for co in ['az', 'el']]
sched_tab = tabulate(sched_data, headers=["t [s]"] + rx_head + ['dwell'])

print(sched_tab)


data = scheduler.observe_passes(passes, orbit = orb, diameter = 0.1)


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

print(data)

for pi in range(len(passes[0][0])):
    dat = data[0][0][pi]
    if dat is not None:
        axes[0][0].plot(passes[0][0][pi].enu[0][0,:], passes[0][0][pi].enu[0][1,:], passes[0][0][pi].enu[0][2,:], '-', label=f'pass-{pi}')
        axes[0][1].plot(dat['t']/3600.0, dat['range'], '-', label=f'pass-{pi}')
        axes[1][0].plot(dat['t']/3600.0, dat['range_rate'], '-', label=f'pass-{pi}')
        axes[1][1].plot(dat['t']/3600.0, 10*np.log10(dat['snr']), '-', label=f'pass-{pi}')

axes[0][1].legend()
plt.show()