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


radar = sorts.radars.tsdr


poptions = dict(
    settings = dict(
        in_frame='GCRS',
        out_frame='ITRS',
    ),
)
obj = sorts.SpaceObject(
    sorts.propagator.SGP4,
    propagator_options = poptions,
    a = 7200e3, 
    e = 0.01, 
    i = 75, 
    raan = 79,
    aop = 0,
    mu0 = 60,
    epoch = Time(53005.0, format='mjd'),
    parameters = dict(
        d = 1.0,
    ),
    oid = 1,
)
t = np.arange(0, 3600*24.0, 10.0)

states = obj.get_state(t)
passes = radar.find_passes(t, states, cache_data = True)

#pick first pass
passes[0][0] = passes[0][0][:1]
inds = passes[0][0][0].inds

track = sorts.controller.Tracker(radar=radar, t0=0, t=t[inds], dwell=0.1, ecefs=states[:3,inds])


#lets make the radar have multiple frequencies
for st in radar.tx + radar.rx:
    st.beam.frequency = [233e6, 133e6]

#now if we look at the tx beam
#This is the shape and names of the parameters
print(f'beam.shape = {radar.tx[0].beam.shape}')
print(f'beam.parameters = {radar.tx[0].beam.parameters}')

k = np.array([0,0,1], dtype=np.float64)

#These parameters are now selectable during gain calculation using the syntax of pyant
print(f'G @ F_1 vs F_2 = {radar.tx[0].beam.gain(k, ind=(0,0))} vs {radar.tx[0].beam.gain(k, ind=(0,1))} ')
print(f'G @ F_1 vs F_2 = {radar.tx[0].beam.gain(k, ind=dict(frequency=0))} vs {radar.tx[0].beam.gain(k, ind=dict(frequency=1))} ')

class Tracking(sorts.scheduler.StaticList, sorts.scheduler.ObservedParameters):

    def get_beam_gain_and_wavelength(self, beam, enu, meta):
        '''Now we have to define how to select the gain
        '''
        if 'frequency' in beam.parameters:
            size = beam.shape[beam.parameters.index('frequency')]
            if size is not None:
                g = [
                    beam.gain(enu, ind={'frequency': pi})
                    for pi in range(size)
                ]
                ind = np.argmax(g)
                return g[ind], beam.wavelength[ind]

        return beam.gain(enu), beam.wavelength


scheduler = Tracking(
    radar = radar, 
    controllers = [track], 
)


data = scheduler.observe_passes(passes, space_object=obj, snr_limit=True)


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

ps = passes[0][0][0]
zang = ps.zenith_angle()

axes[0][0].plot(ps.enu[0][0,:], ps.enu[0][1,:], ps.enu[0][2,:], '-')
axes[0][0].set_xlabel('East-North-Up coordinates')

axes[0][1].plot((ps.t - ps.start())/3600.0, zang[0], '-')
axes[0][1].set_xlabel('Time past epoch [h]')
axes[0][1].set_ylabel('Zenith angle from TX [deg]')

axes[1][0].plot((ps.t - ps.start())/3600.0, ps.range()[0]*1e-3, '-')
axes[1][0].set_xlabel('Time past epoch [h]')
axes[1][0].set_ylabel('Range from TX [km]')

axes[1][1].plot((data[0][0][0]['t'] - ps.start())/3600.0, 10*np.log10(data[0][0][0]['snr']), '.', label='observed-pass')
axes[1][1].set_xlabel('Time past epoch [h]')
axes[1][1].set_ylabel('SNR [dB]')

axes[1][1].legend()
plt.show()
