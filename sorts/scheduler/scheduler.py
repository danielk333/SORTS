#!/usr/bin/env python

'''Scheduler

'''
#Python standard import
from abc import ABC, abstractmethod


#Third party import
import numpy as np

#Local import


class Scheduler(ABC):
    '''A Scheduler for executing time-slices of different radar controllers.
    '''

    def __init__(self, radar, profiler=None, logger=None):
        self.radar = radar
        self.logger = logger
        self.profiler = profiler


    @abstractmethod
    def update(self, *args, **kwargs):
        '''Update the scheduler information.
        '''
        pass


    @abstractmethod
    def get_controllers(self):
        '''This should init a list of controllers and set the `t` (global time) and `t0` (global time reference for controller) variables on them for their individual time samplings.
        '''
        pass


    def generate_schedule(self, t, generator):
        '''Takes times and a corresponding generator that returns radar instances to generate a radar schedule.
        '''
        raise NotImplementedError()


    def calculate_observation(self, txrx_pass, t, generator, **kwargs):
        '''Takes a pass over a tx-rx pair and the corresponding evaluated times and generator that returns radar instances to generate a set of observed data.
        '''
        raise NotImplementedError()


    @staticmethod
    def chain_generators(generators):
        for generator in generators: 
            yield from generator


    def schedule(self):
        ctrls = self.get_controllers()
        times = np.concatenate([c.t for c in ctrls], axis=0)
        sched = Scheduler.chain_generators([c.run() for c in ctrls])
        return self.generate_schedule(times, sched)


    def __call__(self, start, stop):
        ctrls = self.get_controllers()

        check_t = lambda c: np.logical_and(c.t >= start, c.t <= stop)

        ctrls = [c for c in ctrls if np.any(check_t(c))]
        if len(ctrls) == 0:
            if self.logger is not None:
                self.logger.debug(f'Scheduler:__call__:No radar events found between {start:.1f} and {stop:.1f}')
            return None, None
        else:
            t = np.concatenate([c.t[check_t(c)] for c in ctrls], axis=0)
            if self.logger is not None:
                self.logger.debug(f'Scheduler:__call__:{len(t)} events found between {start:.1f} and {stop:.1f}')
            
            return t, Scheduler.chain_generators([c(c.t[check_t(c)] - c.t0) for c in ctrls])


    def observe_passes(self, passes, **kwargs):
        data = []
        for txi in range(len(passes)):
            data.append([])
            for rxi in range(len(passes[txi])):
                data[-1].append([])
                for ps in passes[txi][rxi]:
                    t, generator = self(ps.start(), ps.end())
                    if generator is not None:
                        pass_data = self.calculate_observation(ps, t, generator, **kwargs)
                    else:
                        pass_data = None
                    data[-1][-1].append(pass_data)
        return data
