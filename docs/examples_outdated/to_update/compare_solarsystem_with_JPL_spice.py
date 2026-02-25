#!/usr/bin/env python

'''
Compare solarsystem body state with the python JPL SPICE implementation
=========================================================================
'''
import configparser
import pathlib

import numpy as np
from astropy.time import Time, TimeDelta
import spiceypy as spice

import sorts


try:
    base_pth = pathlib.Path(__file__).parents[1].resolve()
except NameError:
    base_pth = pathlib.Path('.').parents[1].resolve()

config = configparser.ConfigParser(interpolation=None)
config.read([base_pth / 'example_config.conf'])
kernel = config.get('compare_solarsystem_with_JPL_spice.py', 'kernel')


epoch = Time(2001, scale='utc', format='jyear')

spice.furnsh(kernel)
state, lightTime = spice.spkezr(
    'EARTH',
    (epoch.tdb - Time('J2000', scale='tdb')).sec,
    'J2000',
    'NONE',
    '0',
)
state *= 1e3

states = sorts.frames.get_solarsystem_body_states(
    bodies = ['Sun', 'Earth'], 
    epoch = epoch, 
    kernel = kernel,
)

print(f'Earth jplephem: {states["Earth"]}')
print(f'Earth SPICE   : {state}')
print(f'Diff pos      : {states["Earth"][:3]-state[:3]}')
print(f'Diff-pos-norm : {np.linalg.norm(states["Earth"][:3]-state[:3])}')
print(f'Diff vel      : {states["Earth"][3:]-state[3:]}')
print(f'Diff-vel-norm : {np.linalg.norm(states["Earth"][3:]-state[3:])}')
print('\n')

state, lightTime = spice.spkezr(
    'EARTH',
    (epoch.tdb - Time('J2000', scale='tdb')).sec,
    'J2000',
    'NONE',
    'SUN',
)
state *= 1e3

states['Earth'] -= states['Sun']

print('-- HCRS --')
print(f'Earth jplephem: {states["Earth"]}')
print(f'Earth SPICE   : {state}')
print(f'Diff pos      : {states["Earth"][:3]-state[:3]}')
print(f'Diff-pos-norm : {np.linalg.norm(states["Earth"][:3]-state[:3])}')
print(f'Diff vel      : {states["Earth"][3:]-state[3:]}')
print(f'Diff-vel-norm : {np.linalg.norm(states["Earth"][3:]-state[3:])}')
print('\n')

#compare manual transformation between GCRS and HCRS to astropy conversion
state0_GCRS = np.array([6300e3, 0, 0, 20e3, 0, 0])
state0_HCRS = sorts.frames.convert(
    epoch, 
    state0_GCRS, 
    in_frame='GCRS', 
    out_frame='HCRS',
)

h_states = sorts.frames.get_solarsystem_body_states(
    bodies = ['Sun', 'Earth'], 
    epoch = epoch, 
    kernel = kernel,
)

state1_HCRS = state0_GCRS + (h_states['Earth'] - h_states['Sun'])

print(f'State0 HCRS   : {state0_HCRS}')
print(f'State1 HCRS   : {state1_HCRS}')
print(f'Diff pos      : {state0_HCRS[:3]-state1_HCRS[:3]}')
print(f'Diff-pos-norm : {np.linalg.norm(state0_HCRS[:3]-state1_HCRS[:3])}')
print(f'Diff vel      : {state0_HCRS[3:]-state1_HCRS[3:]}')
print(f'Diff-vel-norm : {np.linalg.norm(state0_HCRS[3:]-state1_HCRS[3:])}')
