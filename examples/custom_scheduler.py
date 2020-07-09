#!/usr/bin/env python

'''

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import pyorb

import sorts
from sorts.controller import Tracker
import sorts
eiscat3d = sorts.radars.eiscat3d
from sorts import Scheduler

from sorts.propagator import SGP4
Prop_cls = SGP4
Prop_opts = dict(
    settings = dict(
        out_frame='ITRF',
    ),
)
prop = Prop_cls(**Prop_opts)


class MyScheduler(Scheduler):
    '''
    '''

    def __init__(self, radar, propagator):
        super().__init__(radar)
        self.propagator = propagator
        self.orbits = None


    def update(self, orbits):
        '''Update the scheduler information.
        '''
        self.orbits = orbits


    def get_controllers(self):
        '''This should init all controllers and return a list of them.
        '''

        tv = [np.linspace(x*20,x*20+10,num=5) for x in range(len(self.orbits))]

        ctrls = []
        for ind in range(len(self.orbits)):
            states = self.propagator.propagate(tv[ind], self.orbits.cartesian[:,ind], orb.epoch, A=1.0, C_R = 1.0, C_D = 1.0)

            ctrl = Tracker(radar = self.radar, t=tv[ind], ecefs = states[:3,:])
            ctrls.append(ctrl)
        
        return ctrls


    def generate_schedule(self, t, generator):
        data = np.empty((len(t),len(self.radar.rx)*2+2), dtype=np.float64)
        data[:,0] = t
        data[:,len(self.radar.rx)*2+1] = 0.2
        for ind,radar in enumerate(generator):
            for ri, rx in enumerate(radar.rx):
                data[ind,1+ri*2] = rx.beam.azimuth
                data[ind,2+ri*2] = rx.beam.elevation
        return data


orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees=True, num=3, a=6700e3, e=0, i=75, omega=0, Omega=np.linspace(79,82,num=3), anom=72, epoch=53005)
print(orb)

e3d = MyScheduler(radar = eiscat3d, propagator=prop)

e3d.update(orb)
data = e3d.schedule()



fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(121)
for i in range(3):
    ax.plot(data[:,0], data[:,1+i*2], '.', label=f'RX{i}')
ax.legend()
ax.set_ylabel('Azimuth')

ax = fig.add_subplot(122)
for i in range(3):
    ax.plot(data[:,0], data[:,2+i*2], '.', label=f'RX{i}')
ax.legend()
ax.set_ylabel('Elevation')


plt.show()



# fig, ax =plt.subplots()
# collabel=("Time [s]",)
# for i in range(3):
#     collabel += (f'RX{i} Az [deg]', f'RX{i} El [deg]')
# collabel += ('Dwell [s]',)

# ax.axis('tight')
# ax.axis('off')
# table = ax.table(cellText=data,colLabels=collabel,loc='center')
# table.set_fontsize(22)

# plt.show()
