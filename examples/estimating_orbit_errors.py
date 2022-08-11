#!/usr/bin/env python

'''
=====================================
Estimating orbit determination errors
=====================================

This example computes the measurement jacobian and propagates measurement errors
to calculate orbit determination errors. 
'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from tabulate import tabulate

import pyorb

from sorts.targets import propagator
from sorts.radar import measurement_errors
import sorts

radar = sorts.radars.eiscat3d

try:
    pth = pathlib.Path(__file__).parent / 'data'
except NameError:
    pth = pathlib.Path('.').parent / 'data'

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
    propagator.SGP4,
    propagator_options = dict(
        settings = dict(
            in_frame='TEME', 
            out_frame='ITRS',
        ),
    ),
    state=orb,
    epoch=Time(53005.0, format='mjd', scale='utc'),
    parameters = dict(
        A = 1.0,
    )
)

t = np.arange(0.0, end_t, dt)

print(f'Orbit:\n{str(orb)}')
print(f'Temporal points: {len(t)}')

states = obj.get_state(t)
passes = radar.find_passes(t, states, cache_data=True)

#Create a list the same pass at the other rx stations
#in this example we know that its index 0 at all of them and that it was tri-static
#(otherwise it is simple to find the other passes from this structure)
rx_passes = [p_tx0_rx[0] for p_tx0_rx in passes[0]]

#choose the tx0-rx0 one
ps = rx_passes[0]

#Measure 10 points along pass
use_inds = np.arange(0,len(ps.inds),len(ps.inds)//10)

#Create a radar controller to track the object
track = sorts.controllers.Tracker()
track.meta['target'] = 'Cool object 1'
controls = track.generate_controls(t[ps.inds[use_inds]], radar, t[ps.inds[use_inds]], states[:3,ps.inds[use_inds]], t_slice=dt)
radar_states = radar.control(controls)

p = sorts.Profiler()
p.start('total')

#Now we load the error model
print(f'\nUsing "{pth}" as cache for LinearizedCoded errors.')
err = measurement_errors.LinearizedCodedIonospheric(radar.tx[0], seed=123, cache_folder=pth)

variables = ['x','y','z','vx','vy','vz','A']
deltas = [1e-4]*3 + [1e-6]*3 + [1e-2]

#observe one pass from all rx stations, including measurement Jacobian
for rxi in range(len(radar.rx)):

    #the Jacobian is stacked as [r_measurements, v_measurements]^T so we stack the measurement covariance equally
    data, J_rx = radar.measurement_class.compute_measurement_jacobian(
        rx_passes[rxi], 
        radar_states,
        obj, 
        radar,
        variables, 
        deltas, 
        snr_limit=True,
        exact=True,
        transforms = {
            'A': (lambda A: np.log10(A), lambda Ainv: 10.0**Ainv),
        },
        parallelization=False,
        n_processes=16,
        profiler=p,
    )

    #now we get the expected standard deviations
    r_stds_tx = err.range_std(data["measurements"]['range'], data["measurements"]['snr'])
    v_stds_tx = err.range_rate_std(data["measurements"]['snr'])

    #Assume uncorrelated errors = diagonal covariance matrix
    Sigma_m_diag_tx = np.r_[r_stds_tx**2, v_stds_tx**2]

    #we simply append the results on top of each other for each station
    if rxi > 0:
        J = np.append(J, J_rx, axis=0)
        Sigma_m_diag = np.append(Sigma_m_diag, Sigma_m_diag_tx, axis=0)
    else:
        J = J_rx
        Sigma_m_diag = Sigma_m_diag_tx

    print(f'Range errors std [m] (rx={rxi}):')
    print(r_stds_tx)
    print(f'Velocity errors std [m/s] (rx={rxi}):')
    print(v_stds_tx)


#Add a prior to the Area
Sigma_p_inv = np.zeros((len(variables), len(variables)), dtype=np.float64)
Sigma_p_inv[-1,-1] = 1.0/0.59**2

#diagonal matrix inverse is just element wise inverse of the diagonal
Sigma_m_inv = np.diag(1.0/Sigma_m_diag)

#For a thorough derivation of this formula:
#see Fisher Information Matrix of a MLE with Gaussian errors and a Linearized measurement model
Sigma_orb = np.linalg.inv(np.transpose(J) @ Sigma_m_inv @ J + Sigma_p_inv)


print('Measurement Jacobian size')
print(J.shape)

p.stop('total')
print('\n'+p.fmt(normalize='total'))

print(f'\nLinear orbit estimator covariance [SI-units] (shape={Sigma_orb.shape}):')

header = ['']+variables
header[-1] = 'log10(A)'

list_sig = (Sigma_orb).tolist()
list_sig = [[var] + row for row,var in zip(list_sig, header[1:])]

print(tabulate(list_sig, header, tablefmt="simple"))

ax = sorts.plotting.local_passes([ps])
plt.show()