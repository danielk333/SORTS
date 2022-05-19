#!/usr/bin/env python

'''
Profiling Orekit
======================

'''
import pathlib

import numpy as np

from sorts.common.profiling import Profiler
from sorts.targets.propagator import Orekit
import pyorb

p = Profiler()
p.start('total')

try:
    pth = pathlib.Path(__file__).parent.resolve()
except NameError:
    pth = pathlib.Path('.').parent.resolve()
pth = pth / 'data' / 'orekit-data-master.zip'


if not pth.is_file():
    Orekit.download_quickstart_data(pth, verbose=True)

prop = Orekit(
    orekit_data = pth, 
    settings=dict(
        in_frame='GCRS',
        out_frame='ITRS',
        drag_force = False,
        radiation_pressure = False,
    ),
    profiler = p,
)

print(prop)

orb0 = pyorb.Orbit(M0=pyorb.M_earth, a=7e6, e=0, i=0, omega=0, Omega=0, anom=0)
print(orb0)
state0 = orb0.cartesian.flatten()
t = np.linspace(0, 3600*24.0, num=5000)
mjd0 = 53005

states = prop.propagate(t, state0, mjd0, A=1.0, C_R = 1.0, C_D = 1.0)


p.stop('total')

print(p)
print(p.fmt(timedelta=True))
print(p.fmt(normalize='total'))

print('\nEnable Drag Force \n')

p2 = Profiler()
p2.start('total')

prop = Orekit(
    orekit_data = pth, 
    settings=dict(
        in_frame='GCRS',
        out_frame='ITRS',
        drag_force = True,
        radiation_pressure = False,
    ),
    profiler = p2,
)
states = prop.propagate(t, state0, mjd0, A=1.0, C_R = 1.0, C_D = 1.0)

p2.stop('total')
print(p2.fmt(normalize='total'))
