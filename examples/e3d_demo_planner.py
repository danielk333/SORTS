#!/usr/bin/env python

'''
E3D Demonstrator SST planner
================================

'''
import pathlib

import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from mpl_toolkits.mplot3d import Axes3D
from astropy.time import Time, TimeDelta

import sorts

from sorts.scheduler import Tracking, ObservedParameters


# The Tracking scheduler takes in a series of SpaceObjects and finds all the passes of the objects 
#  over the radar when "update" is called. Then "set_measurements", which is an abstract method,
#  is used to determine when along those passes measurements should be done
# 
# ObservedParameters implements radar observation of space objects based on the radar schedule
#  and calculates the observed parameters (like range, range rate, RCS, SNR, ..)
#  and can be used in case we want to predict what data we will measure
#
# The generate_schedule is not implemented and needs to be defined to generate a schedule output
#  as the standard format for outputting a radar schedule in SORTS is to have a list of "radar"
#  instance with the exact configuration of the radar for each radar action
#
class ObservedTracking(Tracking, ObservedParameters):
    
    def set_measurements(self):
        dw = self.controller_args['dwell']

        #we probably need to make sure we do not have overlapping measurements
        #this is a very "stupid" scheduler but we can do at least that!
        #So create a vector of all scheduled measurements
        t_all = []

        for ind, so in enumerate(self.space_objects):
            #This is a list of all passes times
            t_vec = []
            
            for txi in range(len(self.radar.tx)):
                for rxi in range(len(self.radar.rx)):
                    for ps in self.passes[ind][txi][rxi]:
                        #lets just measure it all! From rise to fall
                        __t = np.arange(ps.start(), ps.end(), dw)

                        #Check for overlap

                        #to keep from this pass
                        t_keep = np.full(__t.shape, True, dtype=np.bool)
                        #to remove (index form) from all previous scheduled
                        t_all_del = []

                        #this creates a matrix of all possible time differences
                        t_diff = np.array(t_all)[:,None] - __t[None,:]

                        #find the ones that overlap with previously selected measurements
                        inds = np.argwhere(np.logical_and(t_diff <= 0, t_diff >= -dw ))

                        #just keep every other, so we are "fair"
                        first_one = True
                        for bad_samp in inds:
                            if first_one:
                                t_keep[bad_samp[1]] = False
                            else:
                                t_all_del.append(bad_samp[0])
                            first_one = not first_one

                        __t = __t[t_keep]

                        #slow and ugly but does the job (filter away measurements)
                        t_all = [t_all[x] for x in range(len(t_all)) if x not in t_all_del]

                        t_vec += [__t]
                        t_all += __t.tolist()

            if self.logger is not None:
                self.logger.info(f'Propagating {sum(len(t) for t in t_vec)} measurement states for object {ind}')

            #epoch difference
            dt = (self.space_objects[ind].epoch - self.epoch).to_value('sec')

            if self.collect_passes:
                t_vec = np.concatenate(t_vec)

                self.states[ind] = so.get_state(t_vec - dt)
                self.states_t[ind] = t
            else:
                self.states[ind] = [so.get_state(t - dt) for t in t_vec]
                self.states_t[ind] = t_vec

    def generate_schedule(self, t, generator, group_target=False):
        data = np.empty((len(t),len(self.radar.tx)*2+len(self.radar.rx)*3+1), dtype=np.float64)

        #here we get a time vector of radar events and the generator that gives the "radar" and meta data for that event
        #Use that to create a schedule table

        all_targets = dict()

        data[:,0] = t
        targets = []
        experiment = []
        passid = []
        for ind,mrad in enumerate(generator):
            radar, meta = mrad
            targets.append(meta['target'])
            
            if meta['target'] in all_targets:
                all_targets[meta['target']] += [ind]
            else:
                all_targets[meta['target']] = [ind]

            experiment.append('SST')
            passid.append(meta['pass'])

            for ti, tx in enumerate(radar.tx):
                data[ind,1+ti*2] = tx.beam.azimuth
                data[ind,2+ti*2] = tx.beam.elevation

            for ri, rx in enumerate(radar.rx):
                data[ind,len(radar.tx)*2+1+ri*3] = rx.beam.azimuth
                data[ind,len(radar.tx)*2+2+ri*3] = rx.beam.elevation
                data[ind,len(radar.tx)*2+3+ri*3] = rx.pointing_range*1e-3 #to km

        data = data.T.tolist() + [experiment, targets, passid]
        data = list(map(list, zip(*data)))

        if group_target:
            #Create a dict of tables instead
            data_ = dict()
            for key in all_targets:
                for ind in all_targets[key]:
                    if key in data_:
                        data_[key] += [data[ind]]
                    else:
                        data_[key] = [data[ind]]
        else:
            data_ = data

        return data_



######## RUNNING ########

from sorts.population import tle_catalog

e3d_demo = sorts.radars.eiscat3d_demonstrator_interp
#############
# CHOOSE OBJECTS
#############

objects = [ #NORAD ID
    27386, #Envisat
    35227,
    35245,
]
epoch = Time('2020-09-08 00:24:51.759', format='iso', scale='utc')
t_start = 0.0
t_end = 12.0*3600.0 #end time of tracking scheduling
t_step = 10.0 #time step for finding passes
dwell = 10.0 #the time between re-pointing beam, i.e. "radar actions" or "time slices"

profiler = sorts.profiling.Profiler()
logger = sorts.profiling.get_logger()

try:
    pth = pathlib.Path(__file__).parent / 'data' / 'space_track_tle.txt'
except NameError:
    import os
    pth = 'data' + os.path.sep + 'space_track_tle.txt'

pop = tle_catalog(pth, kepler=True)

pop.propagator_options['settings']['out_frame'] = 'ITRS' #output states in ECEF

#Get the space objects to track
space_objects = []
for obj in objects:
    ind = np.argwhere(pop.data['oid'] == obj)
    if len(ind) > 0:
        space_objects.append(pop.get_object(ind[0]))

logger.always(f'Found {len(space_objects)} objects to track')

#Initialize the scheduler
scheduler = ObservedTracking(
    radar = e3d_demo, 
    epoch = epoch,
    space_objects = space_objects, 
    end_time = t_end, 
    start_time = t_start, 
    controller_args = dict(return_copy=True, dwell=dwell),
    max_dpos = 1e3,
    profiler = profiler, 
    logger = logger,
    use_pass_states = False,
)

#update the passes
scheduler.update()

#set the measurements using the current passes
scheduler.set_measurements()

#Generate the schedule, grouped by target
grouped_data = scheduler.schedule(group_target=True)

for key in grouped_data:
    pass_id = np.array([x[-1] for x in grouped_data[key]], dtype=np.int) #pass index is last variable
    passes = np.unique(pass_id)

    tv = np.array([x[0] for x in grouped_data[key]]) #we put t at index 0
    az = np.array([x[1] for x in grouped_data[key]]) #we put az at index 1
    el = np.array([x[2] for x in grouped_data[key]]) #and el at index 2

    fig, ax = plt.subplots(1,1)
    for pi, ps in enumerate(passes):
        ax = sorts.plotting.local_tracking(
            az[pass_id == ps], 
            el[pass_id == ps], 
            ax=ax, 
            t=epoch + TimeDelta(tv, format='sec'),
            add_track = pi > 0, #if there are more then one, dont redraw all the extra, just add the track
        )
    ax.set_title(key)

#Generate a combined schedule
data = scheduler.schedule()

#Format and print schedule
rx_head = [f'TX{i}-{co}' for i in range(len(e3d_demo.tx)) for co in ['az [deg]', 'el [deg]']]
rx_head += [f'RX{i}-{co}' for i in range(len(e3d_demo.rx)) for co in ['az [deg]', 'el [deg]', 'r [km]']]
sched_tab = tabulate(data, headers=["t [s]"] + rx_head + ['Experiment', 'Target', 'Pass'])

print(sched_tab)

#Plot radar pointing diagram
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(211)
ax.plot([x[0]/3600.0 for x in data], [x[1] for x in data], ".b")
ax.set_xlabel('Time [h]')
ax.set_ylabel('TX Azimuth [deg]')

ax = fig.add_subplot(212)
ax.plot([x[0]/3600.0 for x in data], [x[2] for x in data], ".b")
ax.set_xlabel('Time [h]')
ax.set_ylabel('TX Elevation [deg]')

plt.show()
