#!/usr/bin/env python

'''This module is used to define the radar controller

'''
#Python standard import
from abc import ABC, abstractmethod
import copy

#Third party import
import numpy as np

#Local import


class RadarController(ABC):
    '''
        Implements the basic structure of a radar controller.
    '''
    
    # TODO : do I need to add a static meta field for t_slice ?

    META_FIELDS = [
        'controller_type',
    ]
    
    def __init__(self, profiler=None, logger=None):
        self.logger = logger
        self.profiler = profiler
    
        # set controller metadata        
        self.meta = dict()
        self.meta['controller_type'] = self.__class__
    
    @abstractmethod
    def generate_controls(self, t, **kwargs):
        '''
            Returns a dictionnary containing the RADAR controls  

            Those controls are generated independantly of the ability of the radar to achieve them.
        '''
        pass