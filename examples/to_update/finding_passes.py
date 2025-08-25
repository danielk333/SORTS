#!/usr/bin/env python

'''
Finding passes over sites
================================

'''

import numpy as np
import matplotlib.pyplot as plt
import pyorb

import sorts
eiscat3d = sorts.radars.eiscat3d

from sorts.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)
prop = Prop_cls(**Prop_opts)


orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees=True, a=7200e3, e=0.1, i=75, omega=0, Omega=79, anom=72, epoch=53005.0)
print(orb)

t = sorts.equidistant_sampling(
    orbit = orb, 
    start_t = 0, 
    end_t = 3600*24*1, 
    max_dpos=1e4,
)

print(f'Temporal points: {len(t)}')


states = prop.propagate(t, orb.cartesian[:,0], orb.epoch, A=1.0, C_R = 1.0, C_D = 1.0)

# fig = plt.figure(figsize=(15,15))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(states[0,:], states[1,:], states[2,:],"-b")
# plt.show()


passes_tx0 = sorts.find_passes(t, states, eiscat3d.tx[0])
print(f'tx-0 passes: {len(passes_tx0)}')

passes_rx1 = sorts.find_passes(t, states, eiscat3d.rx[1])
print(f'rx-1 passes: {len(passes_rx1)}')

#finding simultaneous passes
chtx0 = np.full((len(t),), False, dtype=bool)
chtx0[passes_tx0[0].inds] = True
chrx1 = np.full((len(t),), False, dtype=bool)
chrx1[passes_rx1[0].inds] = True

inds = np.where(np.logical_and(chtx0, chrx1))[0]

print(f'Pass 0 length tx-0: {len(passes_tx0[0].inds)}')
print(f'Pass 0 length rx-1: {len(passes_rx1[0].inds)}')
print(f'Full tx-0 to rx-1 pass: {len(inds)}')

print('Using the predefined paired passes function')

tx0rx1_passes = sorts.find_simultaneous_passes(t, states, [eiscat3d.tx[0], eiscat3d.rx[1]])

print(f'tx-0 and rx-1 passes: {len(tx0rx1_passes)}')
print(f'Full tx-0 to rx-1 pass: {len(tx0rx1_passes[0].inds)}')
