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

#just the first pass
ps = passes[0][0][0]
use_inds = np.arange(0,len(ps.t),len(ps.t)//10)
e3d_tracker = Tracker(radar = eiscat3d, t=ps.t[use_inds], ecefs=states[:3,ps.inds[use_inds]])

class MyStaticList(StaticList):


    def generate_schedule(self, t, generator):
        pass

    def calculate_observation(self, txrx_pass, t, generator, **kwargs):

        snr = np.empty((len(t),), dtype=np.float64)
        enus = txrx_pass.enu
        ranges = txrx_pass.range()
        range_rates = txrx_pass.range_rate()
        zang = txrx_pass.zenith_angle()[0]
        inds = kwargs['inds']

        for ti, radar in enumerate(rgen):
            if radar.tx[txi].enabled and radar.rx[rxi].enabled:
                snr[ti] = sorts.hard_target_snr(
                    radar.tx[txi].beam.gain(enus[0][:,inds[ti]]),
                    radar.rx[rxi].beam.gain(enus[1][:,inds[ti]]),
                    radar.rx[rxi].wavelength,
                    radar.tx[txi].power,
                    ranges[0][inds[ti]],
                    ranges[1][inds[ti]],
                    diameter_m=kwargs['diameter'],
                    bandwidth=radar.tx[txi].coh_int_bandwidth,
                    rx_noise_temp=radar.rx[rxi].noise,
                )
            else:
                snr[ti] = np.nan

        data = dict(
            t = t,
            snr = snr,
            range = ranges[0][inds] + ranges[1][inds],
            range_rate = range_rates[0][inds] + range_rates[1][inds],
            tx_zenith = zang,
        )
        return data


scheduler = MyStaticList(radar = eiscat3d, controllers=[e3d_tracker])

data = scheduler.observe_passes(passes, inds = use_inds, diameter = 0.1)

for pi in range(len(passes[0][0])):
    ps = passes[0][0][pi]
    dat = data[0][0][pi]
    axes[0][0].plot(ps.enu[0][0,:], ps.enu[0][1,:], ps.enu[0][2,:], '-', label=f'pass-{pi}')
    axes[0][1].plot(dat['t']/3600.0, dat['range'], '-', label=f'pass-{pi}')
    axes[1][0].plot(dat['t']/3600.0, dat['range_rate'], '-', label=f'pass-{pi}')
    axes[1][1].plot(dat['t']/3600.0, 10*np.log10(dat['snr']), '-', label=f'pass-{pi}')

axes[0][1].legend()
plt.show()