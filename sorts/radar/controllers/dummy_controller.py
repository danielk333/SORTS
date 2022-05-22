#!/usr/bin/env python

'''This module is used to define the radar controller

'''
#Python standard import
import copy

#Third party import
import numpy as np

#Local import
from . import RadarController

class DummyController(RadarController):
    '''
        This is a dummy controller used for testing only
    '''
    
    META_FIELDS = RadarController.META_FIELDS + [
        'dwell',
        'target',
    ]

    def __init__(self, test_var, profiler=None, logger=None, **kwargs):
        super().__init__(profiler=profiler, logger=logger, **kwargs)
        
        if self.logger is not None:
            self.logger.info(f'Tracker:init')
        
        # setting internal variables 
        self.test_var = test_var
        
        # setting controller metadata        
        self.meta['test_var'] = test_var
    
    def generate_controls(self, t):
       return self.meta