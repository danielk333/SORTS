#!/usr/bin/env python

'''
TEME transformations
=====================

'''

import numpy as np
from astropy.time import Time

import sorts
import sorts.frames as frames


TEME0 = np.array([-7100297.113,-3897715.442,18568433.707,86.771,-3407.231,2961.571])
mjd0 = Time(53005.12, format='mjd')

def print_state(state, newline=False):
    state_ = state.copy()
    if len(state.shape) == 2:
        state_ = state_.reshape(6)
    for i,char in enumerate(['x', 'y', 'z']):
        print(f'{char} = {state_[i]*1e-3:.4e} km  | v{char} = {state_[i+3]*1e-3:.4e} km/s ')
    if newline:
        print('')



ITRS0 = frames.convert(mjd0, TEME0, in_frame='TEME', out_frame='ITRS')
TEME = frames.convert(mjd0, ITRS0, in_frame='ITRS', out_frame='TEME')
ITRS = frames.convert(mjd0, TEME, in_frame='TEME', out_frame='ITRS')

print(f'Transform:')
print_state(TEME0)
print('->')
print_state(ITRS0, newline=True)

print(f'Initial TEME')
print_state(TEME0)
print(f'Recovered TEME')
print_state(TEME)
print(f'Error')
print_state(TEME0-TEME, newline=True)

print(f'Initial ITRS')
print_state(ITRS0)
print(f'Recovered ITRS')
print_state(ITRS)
print(f'Error')
print_state(ITRS0-ITRS, newline=True)

