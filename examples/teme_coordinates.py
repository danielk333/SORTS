#!/usr/bin/env python

'''
TEME transformations
=====================

'''

import numpy as np

import sorts
import sorts.frames as frames

state0 = np.array([-7100297.113,-3897715.442,18568433.707,86.771,-3407.231,2961.571])
mjd0 = 53005.12

def print_state(state, newline=False):
    state_ = state.copy()
    if len(state.shape) == 2:
        state_ = state_.reshape(6)
    for i,char in enumerate(['x', 'y', 'z']):
        print(f'{char} = {state_[i]*1e-3:.4e} km  | v{char} = {state_[i+3]*1e-3:.4e} km/s ')
    if newline:
        print('')

jd = sorts.dates.mjd_to_jd(mjd0)
print(f'Polar motion for Julian date: {jd:1f} days\n')
xp, yp = frames.get_polar_motion(jd)[0,:].tolist()
print(f'xp = {xp:.3e}, yp = {yp:.3e}')

TEME0 = state0.copy()

ITRF0 = frames.TEME_to_ITRF(TEME0, jd, xp, yp)
TEME = frames.ITRF_to_TEME(ITRF0, jd, xp, yp)
ITRF = frames.TEME_to_ITRF(TEME, jd, xp, yp)

print(f'Transform:')
print_state(TEME0)
print('->')
print_state(ITRF0, newline=True)

print(f'Initial TEME')
print_state(TEME0)
print(f'Recovered TEME')
print_state(TEME)
print(f'Error')
print_state(TEME0-TEME, newline=True)

print(f'Initial ITRF')
print_state(ITRF0)
print(f'Recovered ITRF')
print_state(ITRF)
print(f'Error')
print_state(ITRF0-ITRF, newline=True)


#A different implementation of the coordinate transformation

TEME0_v2 = state0.copy()
TEME0_v2 = TEME0_v2.reshape(6,1)

ECEF0 = frames.TEME_to_ECEF(0.0, TEME0_v2[:3,:], TEME0_v2[3:,:], mjd0=mjd0, xp=xp, yp=yp, model='80')
TEME_v2 = frames.ECEF_to_TEME(0.0, ECEF0[:3,:], ECEF0[3:,:], mjd0=mjd0, xp=xp, yp=yp, model='80')
ECEF = frames.TEME_to_ECEF(0.0, TEME_v2[:3,:], TEME_v2[3:,:], mjd0=mjd0, xp=xp, yp=yp, model='80')

print(f'Transform:')
print_state(TEME0_v2)
print('->')
print_state(ECEF0, newline=True)

print(f'Initial ECEF')
print_state(ECEF0)
print(f'Recovered ECEF')
print_state(ECEF)
print(f'Error')
print_state(ECEF0-ECEF, newline=True)

print(f'Initial TEME')
print_state(TEME0_v2)
print(f'Recovered TEME')
print_state(TEME_v2)
print(f'Error')
print_state(TEME0_v2-TEME_v2, newline=True)

print('Implementation difference:')
print_state(ECEF0[:,0] - ITRF0, newline=True)

if sorts.propagator.SGP4 is not None:

    TEME0_sgp4 = state0.copy()
    TLE = frames.TEME_to_TLE(TEME0_sgp4, mjd0, kepler=False, tol=1e-3, tol_v=1e-4)
    TEME_sgp4 = frames.TLE_to_TEME(TLE, mjd0, kepler=False)

    print(f'TEME to SGP4 coordinates (TLE elements):')
    print(f'Transform:')
    print_state(TEME0_sgp4)
    print('->')
    print(TLE)

    print(f'Initial TEME')
    print_state(TEME0_sgp4)
    print(f'Recovered TEME')
    print_state(TEME_sgp4)
    print(f'Error')
    print_state(TEME0_sgp4-TEME_sgp4, newline=True)