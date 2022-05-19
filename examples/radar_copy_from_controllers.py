#!/usr/bin/env python

'''
Radar controller return copy
================================
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pyorb


import sorts
eiscat3d = sorts.radars.eiscat3d
from sorts.radar.controllers import Tracker
from sorts.targets.propagator import SGP4

prop = SGP4(
    settings = dict(
        out_frame='ITRF',
    ),
)

orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees = True, a=6700e3, e=0, i=75, omega=0, Omega=80, anom=72)
t = np.linspace(0,120,num=10)
mjd0 = 53005

states = prop.propagate(t, orb.cartesian[:,0], mjd0, A=1.0, C_R = 1.0, C_D = 1.0)

tcontroller = Tracker(radar=eiscat3d.copy(), t=t, ecefs=states[:3,:], return_copy=True)

eiscat3d.tx[0].beam.sph_point(azimuth=90, elevation=77)

print('Passing a copy to "Tracker" ensures the tracker has its own copy of the radar')
print(f'eiscat3d.tx.pointing         : {eiscat3d.tx[0].beam.pointing}')
print(f'tcontroller.radar.tx.pointing: {tcontroller.radar.tx[0].beam.pointing} \n\n')

print('using "return_copy" ensures that the returned radar from the controller is always its own object')
for radar, meta in tcontroller([t[3]]):

    print(f'tcontroller.radar.tx.pointing: {tcontroller.radar.tx[0].beam.pointing}')
    print(f'radar.tx.pointing            : {radar.tx[0].beam.pointing}')
    print('\n edit tcontroller.radar\n')
    
    tcontroller.radar.tx[0].beam.sph_point(azimuth=120, elevation=44)
    eiscat3d.tx[0].beam.sph_point(azimuth=120, elevation=44)

    print(f'eiscat3d.tx.pointing         : {eiscat3d.tx[0].beam.pointing}')
    print(f'tcontroller.radar.tx.pointing: {tcontroller.radar.tx[0].beam.pointing}')
    print(f'radar.tx.pointing            : {radar.tx[0].beam.pointing}')

