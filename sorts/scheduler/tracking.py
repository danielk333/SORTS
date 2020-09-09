#!/usr/bin/env python

'''

'''
from abc import abstractmethod

import numpy as np
import pyorb

from ..controller import Tracker
from .scheduler import Scheduler
from ..passes import equidistant_sampling, select_simultaneous_passes

class Tracking(Scheduler):
    '''
    '''

    def __init__(self, 
            radar, 
            space_objects, 
            timeslice, 
            allocation, 
            end_time, 
            start_time = 0.0, 
            controller = Tracker,
            controller_args = dict(return_copy=True),
            max_dpos = 1e3,
            profiler = None, 
            logger = None,
            use_pass_states = True,
            calculate_max_snr = False,
        ):
        super().__init__(radar, profiler=profiler, logger=logger)
        self.controller = controller
        self.controller_args = controller_args
        self.space_objects = space_objects
        self.timeslice = timeslice
        self.allocation = allocation
        self.start_time = start_time
        self.end_time = end_time

        self.max_dpos = max_dpos
        self.use_pass_states = use_pass_states
        self.calculate_max_snr = calculate_max_snr

        self.states = [None]*len(space_objects)
        self.states_t = [None]*len(space_objects)
        self.passes = [None]*len(space_objects)

        if self.use_pass_states:
            self.measurements = [None]*len(space_objects)
        else:
            self.measurements = None


    def update(self):
        for ind in range(len(self.space_objects)):
            self.get_passes(ind)


    def get_passes(self, ind):
        t = equidistant_sampling(
            orbit = self.space_objects[ind].orbit, 
            start_t = self.start_time, 
            end_t = self.end_time, 
            max_dpos = self.max_dpos,
        )
        if self.logger is not None:
            self.logger.info(f'Tracking:get_passes(ind={ind}):propagating {len(t)} steps')
        if self.profiler is not None:
            self.profiler.start('Tracking:get_passes:propagating')

        states = self.space_objects[ind].get_state(t)
        if self.use_pass_states:
            self.states[ind] = states
            self.states_t[ind] = t

        if self.profiler is not None:
            self.profiler.stop('Tracking:get_passes:propagating')
        if self.logger is not None:
            self.logger.info(f'Tracking:get_passes(ind={ind}):propagating complete')

        if self.profiler is not None:
            self.profiler.start('Tracking:get_passes:find_passes')
        self.passes[ind] = self.radar.find_passes(t, states, cache_data = True)
        if self.profiler is not None:
            self.profiler.stop('Tracking:get_passes:find_passes')

        #we may need SNR to plan observations
        for txi in range(len(self.radar.tx)):
            for rxi in range(len(self.radar.rx)):
                if self.logger is not None:
                    self.logger.info(f'Tracking:get_passes(ind={ind}):tx{txi}-rx{rxi} {len(self.passes[ind][txi][rxi])} passes')
                if not self.calculate_max_snr:
                    continue
                for ps in self.passes[ind][txi][rxi]:

                    if self.profiler is not None:
                        self.profiler.start('Tracking:get_passes:calculate_max_snr')
                    ps.calculate_max_snr(
                        rx = self.radar.rx[rxi], 
                        tx = self.radar.tx[txi], 
                        diameter = self.space_objects[ind].d,
                    )
                    if self.profiler is not None:
                        self.profiler.stop('Tracking:get_passes:calculate_max_snr')

        return self.passes[ind], states, t


    def get_controllers(self):
        if self.logger is not None:
            self.logger.info(f'Tracking:get_controllers')

        ctrls = []
        for ind in range(len(self.space_objects)):
            if self.use_pass_states:
                minds = self.measurements[ind]
                states = self.states[ind][:3,minds]
                t = self.states_t[ind][minds]
            else:
                states = self.states[ind][:3,:]
                t = self.states_t[ind][:]

            ctrl = self.controller(
                radar = self.radar, 
                t=t, 
                t0=0.0, 
                ecefs = states[:3,:],
                **self.controller_args
            )
            ctrl.meta['target'] = f'Object {self.space_objects[ind].oid}'
            ctrls.append(ctrl)
        
        return ctrls


class PriorityTracking(Tracking):

    def __init__(self, *args, **kwargs):
        self.priority = kwargs.pop('priority', None)

        super().__init__(*args, **kwargs)
        if self.priority is None:
            self.priority = np.ones((len(self.space_objects),), dtype=np.float64)
        

    def calculate_measurements(self):
        total_measurements = int(self.allocation/self.timeslice)
        if not isinstance(self.priority, np.ndarray):
            pri = np.array(self.priority)
            pri = pri/pri.sum()
        else:
            pri = self.priority/self.priority.sum()

        #per object
        measurements = np.floor(pri*total_measurements).astype(np.int64)

        for ind in range(len(self.space_objects)):
            if self.use_pass_states:
                all_inds = []
                for txi in range(len(self.passes[ind])):
                    for rxi in range(len(self.passes[ind][txi])):
                        all_inds += [ps.inds for ps in self.passes[ind][txi][rxi]]
                if len(all_inds) == 0:
                    self.measurements[ind] = np.empty((0,), dtype=np.int64)
                    continue
                all_inds = np.concatenate(all_inds, axis=0)
                all_inds = np.unique(all_inds)
                
                #uniform distribution
                dt = int(len(all_inds)/measurements[ind])
                if dt == 0:
                    dt = 1
                all_inds = all_inds[::dt]
                self.measurements[ind] = all_inds
            else:
                raise NotImplementedError()

            