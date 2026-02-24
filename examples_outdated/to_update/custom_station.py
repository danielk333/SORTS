#!/usr/bin/env python

'''
Custom station type observations
==================================

'''

import types

import numpy as np
import matplotlib.pyplot as plt

import sorts
from sorts.population import tle_catalog

class Camera(sorts.radar.Station):
    def __init__(self, lat, lon, alt):
        super().__init__(lat, lon, alt, 0, None)

    def field_of_view(self, states):
        #Get local East-North-Up
        enu = self.enu(states[:3,:])
        check = enu[1,:] > 0

        return check

#Abisko
station = Camera(
    lat = 68+21/60+20.0/3600,
    lon = 18+49/60+10.5/3600,
    alt = 0.360e3,
)

print(f'lat={station.lat:.2f} deg, lon={station.lon:.2f} deg')

#Some setup for a object to observe

# ENVISAT
tles = [
    (
        '1 27386U 02009A   20312.78435403  .00000027  00000-0  22314-4 0  9994',
        '2 27386  98.1400 312.3173 0001249  93.0784  80.0609 14.37998238979212',
     ),
]

pop = tle_catalog(tles, kepler=False)

pop.propagator_options['settings']['out_frame'] = 'ITRS' #output states in ECEF

obj = pop.get_object(0)
obj.parameters['d'] = 2.0

t = np.arange(0, 3600.0*24.0, 10.0)

states = obj.get_state(t)

passes = sorts.passes.find_simultaneous_passes(
    t, states, 
    [station], 
    cache_data=True,
)

sorts.plotting.local_passes(passes)

plt.show()
