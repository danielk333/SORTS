#!/usr/bin/env python

'''

'''

import numpy as np

from .scheduler import Scheduler


class StaticList(Scheduler):
    '''
    '''

    def __init__(self, radar, controllers):
        super().__init__(radar)
        self.controllers = controllers
 

    def update(self, controllers):
        self.controllers = controllers


    def get_controllers(self):
        return self.controllers


