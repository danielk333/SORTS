#!/usr/bin/env python

'''
E3D Demonstrator SST planner
================================

'''
from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sorts

from sorts import Scheduler
from sorts.propagator import SGP4
from sorts.population import tle_catalog
from sorts.controller import Tracker

radar = sorts.radars.eiscat3d_demonstrator_interp


#############
# CHOOSE OBJECTS
#############

objects = [
    '40838',
    '35227',
    '35245',
]
t_end = 12.0*3600.0 #end time of tracking scheduling
t_step = 10.0

try:
    pth = pathlib.Path(__file__).parent / 'data' / 'space_track_tle.txt'
except NameError:
    import os
    pth = 'data' + os.path.sep + 'space_track_tle.txt'

pop = tle_catalog(pth, kepler=True)


# ObservedParameters implements radar observation of space objects based on the radar schedule
#  and calculates the observed parameters (like range, range rate, RCS, SNR, ..)
#
# The generate_schedule is not implemented and needs to be defined to generate a schedule output
#  as the standard format for outputting a radar schedule in SORTS is to have a list of "radar"
#  instance with the exact configuration of the radar for each radar action
#

class ObservedTracking(ObservedParameters):
    
    def __init__(self, radar, controllers, profiler=None, logger=None, **kwargs):
        super().__init__(
            radar=radar, 
            logger=logger, 
            profiler=profiler,
        )
        self.controllers = controllers


    def update(self, controllers):
        if self.logger is not None:
            self.logger.debug(f'StaticList:update:id(controllers) = {id(controllers)}')
        self.controllers = controllers


    def get_controllers(self):
        return self.controllers



    def generate_schedule(self, t, generator):
        data = np.empty((len(t),len(self.radar.rx)*2+1), dtype=np.float64)
        data[:,0] = t
        names = []
        targets = []
        for ind,mrad in enumerate(generator):
            radar, meta = mrad
            names.append(meta['controller_type'].__name__)
            targets.append(meta['target'])
            for ri, rx in enumerate(radar.rx):
                data[ind,1+ri*2] = rx.beam.azimuth
                data[ind,2+ri*2] = rx.beam.elevation
        data = data.T.tolist() + [names, targets]
        data = list(map(list, zip(*data)))
        return data


controllers = []
for obj in objects:
    pop_id = np.argwhere(pop.data['oid'] == obj)
    space_obj = pop.get_object(pop_id)

    t = np.arange(0, t_end, t_step)
    states = space_obj.get_state(t)

    passes = radar.find_passes(t, states, cache_data = True)

    controllers.append(
        Tracker(
            radar = radar,
            t=,
            ecefs = ,
            meta = dict(
                dwell=0.1,
                target=f'Satnum: {obj}',
            ),
        )
    )



orb = pyorb.Orbit(M0 = pyorb.M_earth, direct_update=True, auto_update=True, degrees=True, num=3, a=6700e3, e=0, i=75, omega=0, Omega=np.linspace(79,82,num=3), anom=72, epoch=53005)
print(orb)

e3d = MyScheduler(radar = eiscat3d, propagator=prop)

e3d.update(orb)
data = e3d.schedule()

rx_head = [f'rx{i} {co}' for i in range(len(eiscat3d.rx)) for co in ['az', 'el']]
sched_tab = tabulate(data, headers=["t [s]"] + rx_head + ['Controller', 'Target'])

print(sched_tab)


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(121)
for i in range(3):
    ax.plot([x[0] for x in data], [x[1+i*2] for x in data], '.', label=f'RX{i}')
ax.legend()
ax.set_ylabel('Azimuth')

ax = fig.add_subplot(122)
for i in range(3):
    ax.plot([x[0] for x in data], [x[2+i*2] for x in data], '.', label=f'RX{i}')
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
