#!/usr/bin/env python

'''
Populations
================================
'''

import numpy as np
import matplotlib.pyplot as plt

from sorts.targets.propagator import SGP4
from sorts.targets import Population
from sorts import plotting

t = np.linspace(0,3600*10,num=2000)

pop = Population(
    fields = ['oid','a','e','i','raan','aop','mu0','mjd0', 'm', 'A', 'C_R', 'C_D'],
    space_object_fields = ['oid', 'm', 'A', 'C_R', 'C_D'],
    state_fields = ['a','e','i','raan','aop','mu0'],
    epoch_field = {'field': 'mjd0', 'format': 'mjd', 'scale': 'utc'},
    propagator = SGP4,
    propagator_options = dict(
        settings = dict(
            out_frame='TEME',
        ),
    )
)
pop.allocate(100)

#for consistency
np.random.seed(120389)

# * 0: oid - Object ID
# * 1: a - Semi-major axis in m
# * 2: e - Eccentricity 
# * 3: i - Inclination in degrees
# * 4: raan - Right Ascension of ascending node in degrees
# * 5: aop - Argument of perihelion in degrees
# * 6: mu0 - Mean anoamly in degrees
# * 7: mjd0 - Epoch of object given in Modified Julian Days
pop['oid'] = np.arange(100)
pop['a'] = np.random.randn(100)*100e3 + 9000e3
pop['e'] = np.random.randn(100)*0.03 + 0.2
pop['i'] = np.random.randn(100)*10 + 69
pop['raan'] = 0
pop['aop'] = 0
pop['mu0'] = np.random.rand(100)*20.0
pop['mjd0'] = 57125.7729
pop['m'] = 0.1
pop['A'] = 1
pop['C_R'] = 0.1
pop['C_D'] = 2.7

plotting.orbits.kepler_scatter(
    pop.get_states(named=False),
    title = "Orbit distribution of Population",
    axis_labels='earth-orbit',
)


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')

for obj in pop:
    states = obj.get_state(t)
    ax.plot(states[0,:], states[1,:], states[2,:],"-b", alpha=0.5)

ax.view_init(-11,17)

plt.show()
    
