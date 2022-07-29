#!/usr/bin/env python

'''
=================================================
Direct simulatiom of atmospheric drag uncertainty 
=================================================

This example performs a direct monte carlo simulation to propagate the uncertainty
from  
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyorb
from astropy.time import Time

from sorts.targets.propagator import SGP4
from sorts.targets import SpaceObject
from sorts.plotting import grid_earth

# step number for the MC simulation
MC_STEP = 100

opts = dict(
    settings = dict(
        out_frame='TEME',
    ),
)

# space object
obj = SpaceObject(
    SGP4,
    propagator_options = opts,
    a = 6800e3, 
    e = 0.0, 
    i = 69, 
    raan = 0, 
    aop = 0, 
    mu0 = 0, 
    epoch = Time(57125.7729, format='mjd'),
    parameters = dict(
        A = 2.0,
    )
)

#change the area every 10 minutes 
dt = 600.0

#propagate for 24h
steps = int(24*3600.0/dt)
states = []

# direct monte carlo step
for mci in range(MC_STEP):
    print(f'Step {mci}/{MC_STEP}')

    mc_obj = obj.copy()
    state = np.empty((6, steps), dtype=np.float64)*np.nan

    # propagate object for each substeps
    for ti in range(steps):
        # update area value accordint to the 20% uncertainty
        mc_obj.parameters['A'] = (1 + np.random.randn(1)[0]*0.2)*obj.parameters['A'] 
        try:
            # propagate
            mc_obj.propagate(dt) 
        except:
            print("could not propagate states.")
            break
        state[:,ti] = mc_obj.orbit.cartesian[:,0]
    print(state[:, -1])
    states += [state]

# plot results
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(111, projection='3d')
fig.suptitle('Anomalous diffusion after 24h with\n 20% normal variation in area')

grid_earth(ax=ax)

fig = plt.figure()
axes = fig.subplots(3, 2)
fig.suptitle('State vectors distribution after 24h with\n 20% normal variation in area')

labels = ["$x$ [$m$]", "$y$ [$m$]", "$z$ [$m$]", "$v_x$ [$m$]", "$v_y$ [$m$]", "$v_z$ [$m$]"]
for mci in range(len(states)):
    ax.plot([states[mci][0,-1]], [states[mci][1,-1]], [states[mci][2,-1]], ".b", alpha=1)

    for i in range(6):
        plot_id_y = int(i/3)
        plot_id_x = i%3
        var1_id = i
        var2_id = (i+1)%6
        axes[plot_id_x][plot_id_y].plot([states[mci][var1_id,-1]], [states[mci][var2_id,-1]], "+b", alpha=1)
        axes[plot_id_x][plot_id_y].set_xlabel(labels[var1_id])
        axes[plot_id_x][plot_id_y].set_ylabel(labels[var2_id])
        axes[plot_id_x][plot_id_y].get_yaxis().get_major_formatter().set_useOffset(False)
        axes[plot_id_x][plot_id_y].get_xaxis().get_major_formatter().set_useOffset(False)
        axes[plot_id_x][plot_id_y].grid()
plt.show()