#!/usr/bin/env python

'''
An example time dynamic scheduler for tracking
================================================
'''

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from astropy.time import Time

import sorts

from sorts.scheduler import PriorityTracking, ObservedParameters
from sorts import SpaceObject
from sorts.profiling import Profiler
from sorts.propagator import SGP4


eiscat3d = sorts.radars.eiscat3d


poptions = dict(
    settings = dict(
        in_frame='GCRS',
        out_frame='ITRS',
    ),
)

epoch = Time(53005.0, format='mjd')

logger = sorts.profiling.get_logger('tracking')

np.random.seed(23847)

def get_objects(mu_std):
    objs = [
        SpaceObject(
            SGP4,
            propagator_options = poptions,
            a = 7200e3, 
            e = 0.01, 
            i = 75, 
            raan = 79,
            aop = 0,
            mu0 = 60 + np.random.randn(1)[0]*mu_std,
            epoch = epoch,
            parameters = dict(
                d = 1.0,
            ),
            oid = 1,
        ),
        SpaceObject(
            SGP4,
            propagator_options = poptions,
            a = 7200e3, 
            e = 0.01, 
            i = 69, 
            raan = 74,
            aop = 0,
            mu0 = 0 + np.random.randn(1)[0]*mu_std,
            epoch = epoch,
            parameters = dict(
                d = 1.0,
            ),
            oid = 42,
        )
    ]
    return objs


true_objects = get_objects(mu_std = 0)
catalogue_objects = get_objects(mu_std = 1.0)

new_request = SpaceObject(
    SGP4,
    propagator_options = poptions,
    a = 7300e3, 
    e = 0.01, 
    i = 77, 
    raan = 74,
    aop = 0,
    mu0 = 12,
    epoch = epoch,
    parameters = dict(
        d = 1.0,
    ),
    oid = 5,
)


class ObservedTracking(PriorityTracking, ObservedParameters):
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



bucket_time = 3600.0
dt = 10.0
max_time = 3600.0*24
#8640 samples

scheduler = ObservedTracking(
    radar = eiscat3d, 
    space_objects = catalogue_objects, 
    timeslice = 0.1, 
    allocation = 0.1*50, 
    end_time = max_time,
    max_dpos = 1e4,
    epoch = epoch,
    priority = [0.2, 1.0],
    logger = logger,
    use_pass_states = True,
    collect_passes = True,
)


def get_table(sch, bucket_index):
    sched_data = sch.schedule(start=bucket_index*bucket_time, stop=(bucket_index+1)*bucket_time)
    rx_head = [f'rx{i} {co}' for i in range(len(sch.radar.rx)) for co in ['az', 'el']]
    return tabulate(sched_data, headers=["t [s]"] + rx_head + ['Controller', 'Target'])


change_occured = [False for obj in scheduler.space_objects]

#set entire interval
scheduler.update()
scheduler.set_measurements()

bi = 0
print_table = True
for i, t in enumerate(np.arange(0, max_time, dt)):
    if t > (bi+1)*bucket_time:
        print_table = True
        bi += 1

    #TODO: whenever we actual pass a set of measurements
    #remove time from the scheduler.allocation time

    #let scheduler know what time it is
    scheduler.start_time = t

    #events occurring
    if i == 360:
        #we get better orbital elements for object 2
        scheduler.space_objects[1] = true_objects[1]
        change_occured[1] = True
    elif i == 2000:
        #someone pays us a lot of money to look at object 1
        scheduler.priority[0] = scheduler.priority[1]*2
        change_occured[0] = True
    elif i == 6222:
        #We get new information about a new object
        scheduler.add_space_object(new_request, priority=1.0)
        change_occured.append(True)

    #recalculate entire rest of the interval, might be inefficient if many changes occur
    for ind, change in enumerate(change_occured):
        if change:
            scheduler.get_passes(ind)
            scheduler.set_measurements()
            print_table = True

    #reset changes tracking
    change_occured = [False for obj in scheduler.space_objects]
    if print_table:
        #print schedule
        print(f'Schedule for bucket {bi} @ t={t/3600.0:.3f} h')
        print(get_table(scheduler, bi))
        print_table = False
