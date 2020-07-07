#!/usr/bin/env python

'''

'''

import numpy as np

from .scheduler import Scheduler


class StaticList(Scheduler):
    '''
    '''

    def __init__(self, radar, controllers, profiler=None, logger=None, **kwargs):
        super().__init__(
            radar=radar, 
            logger=logger, 
            profiler=profiler,
        )
        self.controllers = controllers


    def update(self, controllers):
        if self.logger is not None:
            self.logger.debug(f'StaticList:update:id(controllers) = {id(controllers)}')
        self.controllers = controllers


    def get_controllers(self):
        return self.controllers


