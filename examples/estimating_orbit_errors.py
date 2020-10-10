#!/usr/bin/env python

'''
Estimating orbit determination errors
======================================

'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time

import sorts.propagator as propagators
import sorts.errors as errors
import sorts

radar = sorts.radars.eiscat3d

try:
    pth = pathlib.Path(__file__).parent / 'data'
except NameError:
    import os
    pth = 'data' + os.path.sep

dt = 10.0
end_t = 3600.0*24.0

orb = pyorb.Orbit(
    M0 = pyorb.M_earth, 
    direct_update=True, 
    auto_update=True, 
    degrees=True, 
    a=7200e3, 
    e=0.01, 
    i=75, 
    omega=0, 
    Omega=79, 
    anom=72, 
)
obj = sorts.SpaceObject(
    SGP4,
    propagator_options = dict(
        settings = dict(
            in_frame='TEME', 
            out_frame='ITRS',
        ),
    ),
    state = orb,
    epoch=Time(53005.0, format='mjd', scale='utc'),
    parameters = dict(
        d = 0.2,
    )
)


t = np.arange(0.0, end_t, dt)

print(f'Orbit:\n {str(orb)}')
print(f'Temporal points: {len(t)}')

states = obj.get_state(t)

passes = radar.find_passes(t, states)

#Extract a pass over a tx-rx pair
p_tx0_rx0 = passes[0][0]

#choose the first one
ps = p_tx0_rx0[0]

#Measure 10 points along pass
use_inds = np.arange(0,len(ps.inds),len(ps.inds)//10)

#Create a radar controller to track the object
track = sorts.controller.Tracker(radar = radar, t=t[ps.inds[use_inds]], ecefs=states[:3,ps.inds[use_inds]])
track.meta['target'] = 'Cool object 1'

class Schedule(
        sorts.scheduler.StaticList, 
        sorts.scheduler.ObservedParameters,
    ):
    pass

sched = Schedule(radar = eiscat3d, controllers=[track])

#observe one pass
t, generator = sched(ps.start(), ps.end())
data = sched.calculate_observation(ps, t, generator, space_object=obj, snr_limit=True)


#Now we load the error model
print(f'Using "{pth}" as cache for LinearizedCoded errors.')
err = errors.LinearizedCoded(radar.tx[0], seed=123, cache_folder=pth)

#now we get the expected standard deviations
r_stds = err.range_std(data['snr'])
v_stds = err.range_rate_std(data['snr'])


from pyod import OptimizeLeastSquares, MCMCLeastSquares

import pyorb

prop = SGP4(
    settings=dict(
        in_frame='TEME',
        out_frame='ITRS',
    )
)


variables = ['x', 'y', 'z', 'vx', 'vy', 'vz']
dtype = [(name, 'float64') for name in variables]

state0_named = np.empty((1,), dtype=dtype)
step_arr = np.array([1e3,1e3,1e3,1e1,1e1,1e1], dtype=np.float64)
step = np.empty((1,), dtype=dtype)

for ind, name in enumerate(variables):
    state0_named[name] = state0[ind]
    step[name] = step_arr[ind]


input_data_state = {
    'sources': sources,
    'Model': ,
    'date0': prior_time,
    'params': params,
}

post_init = OptimizeLeastSquares(
    data = input_data_state,
    variables = variables,
    start = state0_named,
    prior = None,
    propagator = prop,
    method = 'Nelder-Mead',
    options = dict(
        maxiter = 10000,
        disp = False,
        xatol = 1e-3,
    ),
)