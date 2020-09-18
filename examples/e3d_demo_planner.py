#!/usr/bin/env python

'''
E3D Demonstrator SST planner
================================

'''
import pathlib

from tabulate import tabulate
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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

        for ind, so in enumerate(self.space_objects):
            t_vec = []
            for txi in range(len(self.radar.tx)):
                for rxi in range(len(self.radar.rx)):
                    for ps in self.passes[ind][txi][rxi]:
                        #lets just measure it all!
                        t_vec += np.arange(ps.start(), ps.end(), dw).tolist()

            #we probably need to make sure we do not have overlapping measurements
            #this is a very "stupid" scheduler
            t = np.array(t_vec)
            t_keep = np.full(t.shape, True, dtype=np.bool)

            for ch_ind in range(ind):
                t_ch_keep = np.full(self.states_t[ch_ind].shape, True, dtype=np.bool)
                #this creates a matrix of all possible time differences
                t_diff = self.states_t[ch_ind][:,None] - t[None,:]

                #find the ones that overlap with previously selected measurements
                inds = np.argwhere(np.logical_and(t_diff <= 0, t_diff >= -dw ))

                #just do every other
                first_one = True
                for bad_samp in inds:
                    if first_one:
                        t_keep[bad_samp[1]] = False
                    else:
                        t_ch_keep[bad_samp[0]] = False

                    first_one = not first_one
                t = t[t_keep]

                self.states_t[ch_ind] = self.states_t[ch_ind][t_ch_keep]
                self.states[ch_ind] = self.states[ch_ind][:,t_ch_keep]

            if self.logger is not None:
                self.logger.info(f'Propagating {len(t)} measurement states for object {ind}')

            self.states[ind] = so.get_state(t)
            self.states_t[ind] = t


    def generate_schedule(self, t, generator):
        data = np.empty((len(t),len(self.radar.tx)*2+len(self.radar.rx)*3+1), dtype=np.float64)

        data[:,0] = t
        targets = []
        experiment = []
        for ind,mrad in enumerate(generator):
            radar, meta = mrad
            targets.append(meta['target'])
            experiment.append('SST')

            for ti, tx in enumerate(radar.tx):
                data[ind,1+ti*2] = tx.beam.azimuth
                data[ind,2+ti*2] = tx.beam.elevation

            for ri, rx in enumerate(radar.rx):
                data[ind,len(radar.tx)*2+1+ri*3] = rx.beam.azimuth
                data[ind,len(radar.tx)*2+2+ri*3] = rx.beam.elevation
                data[ind,len(radar.tx)*2+3+ri*3] = rx.pointing_range*1e-3 #to km

        data = data.T.tolist() + [experiment, targets]
        data = list(map(list, zip(*data)))
        return data




if __name__=='__main__':
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

    space_objects = []
    for obj in objects:
        ind = np.argwhere(pop.data['oid'] == obj)
        if len(ind) > 0:
            space_objects.append(pop.get_object(ind[0]))

    logger.always(f'Found {len(space_objects)} objects to track')

    scheduler = ObservedTracking(
        radar = e3d_demo, 
        space_objects = space_objects, 
        end_time = t_end, 
        start_time = t_start, 
        controller_args = dict(return_copy=True, dwell=dwell),
        max_dpos = 1e3,
        profiler = profiler, 
        logger = logger,
        use_pass_states = False,
    )

    scheduler.update()
    scheduler.set_measurements()

    data = scheduler.schedule()

    rx_head = [f'TX{i}-{co}' for i in range(len(e3d_demo.tx)) for co in ['az [deg]', 'el [deg]']]
    rx_head += [f'RX{i}-{co}' for i in range(len(e3d_demo.rx)) for co in ['az [deg]', 'el [deg]', 'r [km]']]
    sched_tab = tabulate(data, headers=["t [s]"] + rx_head + ['Experiment', 'Target'])

    print(sched_tab)