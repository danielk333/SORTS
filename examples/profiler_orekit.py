#!/usr/bin/env python

'''
Profiling components
======================

'''
import numpy as np

from sorts.profiling import Profiler
from sorts.propagator import Orekit

p = Profiler()

orekit_data = '/home/danielk/IRF/IRF_GITLAB/orekit_build/orekit-data-master.zip'

prop = Orekit(
    orekit_data = orekit_data, 
    settings=dict(
        in_frame='ITRF',
        out_frame='EME',
        drag_force = False,
        radiation_pressure = False,
    ),
    profiler = p,
)

print(prop)

state0 = np.array([-7100297.113,-3897715.442,18568433.707,86.771,-3407.231,2961.571])
t = np.linspace(0,3600*24.0*2,num=5000)
mjd0 = 53005

states = prop.propagate(t, state0, mjd0, A=1.0, C_R = 1.0, C_D = 1.0)

print(p)