#!/usr/bin/env python

'''
Fitting mean elements
======================

'''
import numpy as np
import scipy.optimize as sio
import matplotlib.pyplot as plt

import pyorb

from sorts.propagator import SGP4

#reproducibility
np.random.seed(324245)

prop = SGP4(
    settings = dict(
        in_frame='TEME',
        out_frame='TEME',
    ),
)
std_pos = 1e3 #1km std noise on positions

orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees=True, a=7200e3, e=0.05, i=75, omega=0, Omega=79, anom=72, type='mean')
print(orb)

state0 = orb.cartesian[:,0]
t = np.linspace(0,600.0,num=100)
mjd0 = 53005

params = dict(A=1.0, C_R=1.0, C_D=2.3)

states = prop.propagate(t, state0, mjd0, **params)
noisy_pos = states[:3,:] + np.random.randn(3,len(t))*std_pos

#now for the least squares function to minimize
def lsq(mean_elements):

    states = prop.propagate(t, mean_elements, mjd0, SGP4_mean_elements=True, **params)
    rv_diff = np.linalg.norm(noisy_pos - states[:3,:], axis=0)
    return rv_diff.sum()

#initial guess is just kepler elements
mean0 = orb.kepler[:,0]

#The order is different (and remember its mean anomaly), but we still use SI units
tmp = mean0[4]
mean0[4] = mean0[3]
mean0[3] = tmp

res = sio.minimize(lsq, mean0, method='Nelder-Mead', options={'ftol': 1e-8, 'maxfev': 10000})

final_states = prop.propagate(t, res.x, mjd0, SGP4_mean_elements=True, **params)

print(res.x)
print(orb.kepler)

print(res)

fig = plt.figure(figsize=(15,15))
for i in range(3):
    ax = fig.add_subplot(311 + i)
    ax.plot(t/3600.0, final_states[i,:], '-r', label='Fitted states')
    ax.plot(t/3600.0, noisy_pos[i,:], '.b', label='Measured states')
    ax.plot(t/3600.0, states[i,:], '--g', label='True states')

ax.legend()

fig = plt.figure(figsize=(15,15))
for i in range(3):
    ax = fig.add_subplot(311 + i)
    ax.plot(t/3600.0, final_states[i,:] - states[i,:], '-b', label='True Error')
    ax.plot(t/3600.0, final_states[i,:] - noisy_pos[i,:], '-r', label='Residuals')
ax.legend()

plt.show()