#!/usr/bin/env python

''' Scheduler base class.

'''
#Python standard import
from abc import ABC, abstractmethod


#Third party import
import numpy as np

#Local import


class Scheduler(ABC):
    ''' 
    Used to schedule measurements prior to perfoming the computation of controls by the controllers.
    '''

    def __init__(self, profiler=None, logger=None):
        self.logger = logger
        self.profiler = profiler

        if self.logger is not None:
            self.logger.info("scheduler:init")

    @abstractmethod
    def generate_schedule(self, t):
        ''' 
        Generates the schedule according to the time array t
        '''
        raise NotImplementedError()