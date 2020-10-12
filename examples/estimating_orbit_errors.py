#!/usr/bin/env python

'''
Estimating orbit determination errors
======================================

'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from tabulate import tabulate

import pyorb

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
    propagators.SGP4,
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

#Create a list the same pass at the other rx stations
#in this example we know that its index 0 at all of them and that it was tri-static
#(otherwise it is simple to find the other passes from this structure)
rx_passes = [p_tx0_rx[0] for p_tx0_rx in passes[0]]

#choose the tx0-rx0 one
ps = rx_passes[0]

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

p = sorts.Profiler()
p.start('total')

sched = Schedule(radar = radar, controllers=[track], profiler=p)


try:
    pth = pathlib.Path(__file__).parent / 'data'
except NameError:
    import os
    pth = 'data' + os.path.sep

#Now we load the error model
print(f'\nUsing "{pth}" as cache for LinearizedCoded errors.')
err = errors.LinearizedCoded(radar.tx[0], seed=123, cache_folder=pth)

variables = ['x','y','z','vx','vy','vz']

#observe one pass from all rx stations, including measurement Jacobian


for rxi in range(len(radar.rx)):
    #the Jacobean is stacked as [r_measurements, v_measurements]^T so we stack the measurement covariance equally
    data, J_rx = sched.calculate_observation_jacobian(rx_passes[rxi], space_object=obj, variables=variables, deltas=[1e-4]*3 + [1e-6]*3, snr_limit=True)

    #now we get the expected standard deviations
    r_stds_tx = err.range_std(data['snr'])
    v_stds_tx = err.range_rate_std(data['snr'])

    #Assume uncorrelated errors = diagonal covariance matrix
    Sigma_m_diag_tx = np.r_[r_stds_tx**2, v_stds_tx**2]

    #we simply append the results on top of each other for each station
    if rxi > 0:
        J = np.append(J, J_rx, axis=0)
        Sigma_m_diag = np.append(Sigma_m_diag, Sigma_m_diag_tx, axis=0)
    else:
        J = J_rx
        Sigma_m_diag = Sigma_m_diag_tx
        v_stds = v_stds_tx

    print(f'Range errors std [m] (rx={rxi}):')
    print(r_stds_tx)
    print(f'Velocity errors std [m/s] (rx={rxi}):')
    print(v_stds_tx)


#diagonal matrix inverse is just element wise inverse of the diagonal
Sigma_m_inv = np.diag(1.0/Sigma_m_diag)

Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J)


print('Measurement Jacobian size')
print(J.shape)

p.stop('total')
print('\n'+p.fmt(normalize='total'))

print('\nLinear orbit estimator covariance [km & km/s]:')

list_sig = (Sigma_orb*1e-3).tolist()
list_sig = [[var] + row for row,var in zip(list_sig, variables)]

print(tabulate(list_sig, ['']+variables, tablefmt="simple"))

ax = sorts.plotting.local_passes([ps])
plt.show()