#!/usr/bin/env python

'''
Profiling SGP4
======================

'''
import numpy as np

from sorts.common.profiling import Profiler
from sorts.targets.propagator import SGP4

p = Profiler()

prop = SGP4(
    settings = dict(
        out_frame='TEME',
    ),
    profiler = p,
)

state0 = np.array([-7100297.113,-3897715.442,18568433.707,86.771,-3407.231,2961.571])
t = np.linspace(0,3600*24.0*2,num=5000)
mjd0 = 53005

print(prop)

states = prop.propagate(t, state0, mjd0, A=1.0, C_R = 1.0, C_D = 1.0)

print(p)