#!/usr/bin/env python

'''
=============
Interpolation
=============

This exanple showcases the use of ``sorts`` interpolation algorithms to 
reduce computational overhead by propagating at a lower time-resolution.

It compares the two main interpolation algorithm :
* Legendre8 : 8th degree polynomial interpolation
* Linear : linear interpolation
'''
import numpy as np
import matplotlib.pyplot as plt

from sorts.common import Profiler, interpolation
from sorts.targets.propagator.pysgp4 import SGP4
from sorts.common import interpolation

p = Profiler()

# initializes SGP4 propagator
prop = SGP4(
    settings = dict(
        out_frame='TEME',
    ),
    profiler = p,
)

# initializes initial object states
state0 = np.array([-7100297.113,-3897715.442,18568433.707,86.771,-3407.231,2961.571])
t = np.arange(0.0,360.0+30.0,step=30.0)
mjd0 = 53005

# propagation of states at low-res -> 30s step
states = prop.propagate(t, state0, mjd0, A=1.0, C_R = 1.0, C_D = 1.0)

# interpolation time array at 1s step
t_f = np.arange(0.0,360.0,step=1.0)
interpolator = interpolation.Legendre8(states, t)
finer_states = interpolator.get_state(t_f)
lin_interp = interpolation.Linear(states, t)
lin_states = lin_interp.get_state(t_f)

# plot results and compare linear to legendre8
labels = ["$x$ [$m$]", "$y$ [$m$]", "$z$ [$m$]", "$v_x$ [$m/s$]", "$v_y$ [$m/s$]", "$v_z$ [$m/s$]"]
fig = plt.figure()
fig.suptitle("State interpolation")
for ind in range(3):
    ax = fig.add_subplot(321 + ind)
    ax.plot(t_f, finer_states[ind,:],".r")
    ax.plot(t_f, lin_states[ind,:],".g")
    ax.plot(t, states[ind,:],"xb")
    ax.set_ylabel(labels[ind])
    ax.set_xlabel("$t$ [$s$]")

    ax = fig.add_subplot(324 + ind)
    ax.plot(t_f, finer_states[ind+3,:],".r")
    ax.plot(t_f, lin_states[ind+3,:],".g")
    ax.plot(t, states[ind+3,:],"xb")
    ax.set_ylabel(labels[ind+3])
    ax.set_xlabel("$t$ [$s$]")


# plot residuals
labels = ["$\Delta x$ [$m$]", "$\Delta y$ [$m$]", "$\Delta z$ [$m$]", "$\Delta v_x$ [$m/s$]", "$\Delta v_y$ [$m/s$]", "$\Delta v_z$ [$m/s$]"]
fig = plt.figure()
fig.suptitle("Residuals")
for ind in range(3):
    ax = fig.add_subplot(321 + ind)
    ax.plot(t_f, finer_states[ind,:] - lin_states[ind,:],"-b")
    ax.set_ylabel(labels[ind])
    ax.set_xlabel("$t$ [$s$]")

    ax = fig.add_subplot(324 + ind)
    ax.plot(t_f, finer_states[ind+3,:] - lin_states[ind+3,:],"-b")
    ax.set_ylabel(labels[ind+3])
    ax.set_xlabel("$t$ [$s$]")
plt.show()
