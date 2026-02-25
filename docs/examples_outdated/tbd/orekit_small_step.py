#!/usr/bin/env python

'''
Orekit propagator small time step
====================================
'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sorts.propagator import Orekit

try:
    pth = pathlib.Path(__file__).parent.resolve()
except NameError:
    pth = pathlib.Path('.').parent.resolve()
pth = pth / 'data' / 'orekit-data-master.zip'


if not pth.is_file():
    sorts.propagator.Orekit.download_quickstart_data(pth, verbose=True)

prop = Orekit(
    orekit_data = pth, 
    settings=dict(
        in_frame='Orekit-ITRF',
        out_frame='Orekit-ITRF',
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