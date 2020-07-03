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
        self.event_list = [] #the order of controllers to execute
        self.samplings = [] #a list of numpy arrays of sampling times for schedule generation


    @abstractmethod
    def update(self, *args, **kwargs):
        '''Update the scheduler information.
        '''
        pass


    @abstractmethod
    def get_controller(self, event, t):
        '''This should init a controller and call it to return the generator for that controllers time-slice.
        '''
        pass


    @abstractmethod
    def format_schedule(self, t, generator):
        pass


    @staticmethod
    def chain_generators(generators):
        for generator in generators: 
            yield from generator


    def schedule(self):
        ctrls = [self.get_controller(event, self.samplings[ind]) 
            for ind,event in enumerate(self.event_list)
        ]
        samples = np.concatenate(self.samplings, axis=0)
        sched = Scheduler.chain_generators(ctrls)
        return self.format_schedule(samples, sched)


    def turn_off(self):
        for st in self.radar.tx + self.radar.rx:
            st.enabled = False


    def turn_on(self):
        for st in self.radar.tx + self.radar.rx:
            st.enabled = True