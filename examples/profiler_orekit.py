#!/usr/bin/env python

'''
Profiling Orekit
======================

'''
import pathlib

import numpy as np

from sorts.profiling import Profiler
from sorts.propagator import Orekit

p = Profiler()
p.start('total')

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
        in_frame='ITRS',
        out_frame='GCRS',
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


p.stop('total')

print(p)
print(p.fmt(timedelta=True))
print(p.fmt(normalize='total'))

print('\n Enable Drag Force \n')

p2 = Profiler()
p2.start('total')

prop = Orekit(
    orekit_data = orekit_data, 
    settings=dict(
        in_frame='ITRS',
        out_frame='GCRS',
        drag_force = True,
        radiation_pressure = False,
    ),
    profiler = p2,
)
states = prop.propagate(t, state0, mjd0, A=1.0, C_R = 1.0, C_D = 1.0)

p2.stop('total')
print(p2.fmt(normalize='total'))
