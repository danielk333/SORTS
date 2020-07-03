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

    def __init__(self, radar):
        self.radar = radar


    @abstractmethod
    def update(self, *args, **kwargs):
        '''Update the scheduler information.
        '''
        pass


    @abstractmethod
    def get_controllers(self):
        '''This should init a list of controllers and set the `t` variables on them for their individual time samplings.
        '''
        pass


    @abstractmethod
    def generate_schedule(self, t, generator):
        pass


    @staticmethod
    def chain_generators(generators):
        for generator in generators: 
            yield from generator


    def schedule(self):
        ctrls = self.get_controllers()
        times = np.concatenate([c.t for c in ctrls], axis=0)
        sched = Scheduler.chain_generators([c.run() for c in ctrls])
        return self.generate_schedule(times, sched)


    def turn_off(self):
        for st in self.radar.tx + self.radar.rx:
            st.enabled = False


    def turn_on(self):
        for st in self.radar.tx + self.radar.rx:
            st.enabled = True