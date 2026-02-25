#!/usr/bin/env python

'''
Custom Field of View
======================

'''

import types

import numpy as np
import matplotlib.pyplot as plt

import sorts
from sorts.population import tle_catalog

radar = sorts.radars.eiscat3d_demonstrator_interp

print(f'lat={radar.tx[0].lat:.2f} deg, lon={radar.tx[0].lon:.2f} deg')

#lets define a FOV to only be north looking
def new_fov(self, states):
    #Get local East-North-Up
    enu = self.enu(states[:3,:])
    
    check = enu[1,:] > 0

    return check


#and set that fov for all stations
for st in radar.tx + radar.rx:
    #we need to make sure its actually a method connected to the instance
    st.field_of_view = types.MethodType(new_fov, st)
    

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

passes = radar.find_passes(t, states)

sorts.plotting.local_passes(passes[0][0])

plt.show()
