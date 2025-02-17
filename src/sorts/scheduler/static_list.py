#!/usr/bin/env python

"""Provides a static list of controllers, simplest scheduler.

"""

import logging

from .scheduler import Scheduler

logger = logging.getLogger(__name__)

class StaticList(Scheduler):
    """#TODO: Docstring"""

    def __init__(self, radar, controllers, **kwargs):
        super().__init__(
            radar=radar,
        )
        self.controllers = controllers

    def update(self, controllers):
        logger.debug(f"StaticList:update:id(controllers) = {id(controllers)}")
        self.controllers = controllers

    def get_controllers(self):
        return self.controllers
