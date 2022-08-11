#!/usr/bin/env python

'''
================
Custom Scheduler
================

This short example shows how to define a simple custom scheduler using the predifined :class:`sorts.RadarSchedulerBase` class.
The scheduler implemented in this example generates tracking controls from an object's orbit.
'''
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pyorb

import sorts
from sorts.targets.propagator import SGP4

eiscat3d = sorts.radars.eiscat3d
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)

prop = Prop_cls(**Prop_opts)

class MyScheduler(sorts.RadarSchedulerBase):
    ''' Generates tracking controls from an orbit. '''
    def __init__(self, radar, propagator, t0, scheduler_period):
        super().__init__(radar, t0, scheduler_period)
        self.propagator = propagator

    def run(self, orbits):
        ''' Generation of the control sequence. '''
        tv = [np.linspace(x*20,x*20+10,num=5) for x in range(len(orbits))]

        ctrls = []
        for ind in range(len(orbits)):
            states = self.propagator.propagate(tv[ind], orbits.cartesian[:,ind], orbits.epoch, A=1.0, C_R = 1.0, C_D = 1.0)

            ctrl = sorts.Tracker()

            controls = ctrl.generate_controls(tv[ind], self.radar, tv[ind], states[:3,:], t_slice=0.1, scheduler=self)
            controls.meta['target'] = f'Orbit {ind}'
            controls.meta['controller_type'] = 'Tracker'
            ctrls.append(controls)
        return controls


    def generate_schedule(self, controls):
        ''' Generate radar schedule. '''
        # extract scheduler data
        data = np.empty((controls.n_periods,), dtype=object)
        for period_id in range(controls.n_periods):
            n_points    = len(controls.t[period_id])

            names = np.repeat(controls.meta['controller_type'], n_points)
            targets = np.repeat(controls.meta['target'], n_points)
            data_tmp    = np.ndarray((n_points, len(self.radar.rx)*2 + 1))
            data_tmp[:,0] = controls.t[period_id]
            
            pdirs = controls.get_pdirs(period_id)
            for ri, rx in enumerate(self.radar.rx):
                rx.point_ecef(pdirs["rx"][ri, 0, :, :])
                data_tmp[:,1+ri*2] = rx.beam.azimuth
                data_tmp[:,2+ri*2] = rx.beam.elevation

            data_tmp = data_tmp.T.tolist() + [names, targets]
            data_tmp = list(map(list, zip(*data_tmp)))
            data[period_id] = data_tmp
        return data

# define object orbit
orb = pyorb.Orbit(M0=pyorb.M_earth, direct_update=True, auto_update=True, degrees=True, num=3, a=6700e3, e=0, i=75, omega=0, Omega=np.linspace(79,82,num=3), anom=72, epoch=53005)
print(orb)

# instanciation of the scheduler and generation of the controls
e3d = MyScheduler(eiscat3d, prop, t0=0, scheduler_period=5)
controls = e3d.run(orb)
data = e3d.generate_schedule(controls)

# prints the schedule from the generated controls
print("\nSchedule")
for period_id in range(controls.n_periods):
    print("Period index ", period_id)
    rx_head = [f'rx{i} {co}' for i in range(len(eiscat3d.rx)) for co in ['az', 'el']]
    sched_tab = tabulate(data[period_id], headers=["t [s]"] + rx_head + ['Controller', 'Target'])
    print(sched_tab, "\n")

# plots the azimuth and elevation of the radar as a function of time
fig = plt.figure(figsize=(15,15))
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

for period_id in range(controls.n_periods):
    for i in range(3):
        ax1.plot([x[0] for x in data[period_id]], [x[1+i*2] for x in data[period_id]], '+', label=f'RX{i}')
    for i in range(3):
        ax2.plot([x[0] for x in data[period_id]], [x[2+i*2] for x in data[period_id]], '+', label=f'RX{i}')
    
ax1.legend()
ax1.set_ylabel('Azimuth [$deg$]')
ax1.set_xlabel('$t$ [$s$]')
ax1.grid()
ax2.legend()
ax2.set_ylabel('Elevation [$deg$]')
ax2.set_xlabel('$t$ [$s$]')
ax2.grid()
plt.show()