#!/usr/bin/env python

'''
Orekit propagator small time step
====================================
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sorts.propagator import Orekit

orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'

prop = Orekit(
    orekit_data = orekit_data, 
    settings=dict(
        in_frame='ITRF',
        out_frame='ITRF',
        drag_force = False,
        radiation_pressure = False,
    )
)

print(prop)

state0 = np.array([-7100297.113,-3897715.442,18568433.707,86.771,-3407.231,2961.571])
t = np.arange(100000)*1e-6
mjd0 = 53005

states = prop.propagate(t, state0, mjd0, A=1.0, C_R = 1.0, C_D = 1.0)


fig = plt.figure(figsize=(15,15))
for i in range(3):
    ax = fig.add_subplot(311+i)
    ax.plot(t, states[i,:],"-b")

plt.show()