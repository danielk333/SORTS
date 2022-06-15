import numpy as np
from abc import ABC, abstractmethod

import scipy.constants

from .. import signals
from ..passes import Pass

class Measurement(ABC):
	def __init__(self, radar, logger=None, profiler=None):
		self.radar=radar
        self.logger=logger
        self.profiler=profiler

        if self.logger is not None:
            self.logger.info(f'Measurement:Initialization -> intitializing measurement unit')

	def compute_observation_jacobian(slef):
		pass

	def measure(self):
		pass

	@abstractmethod
	def stop_condition(self, t):
		'''
        Measurement abort/stop condition (i.e. stop time, ...)
        '''
		pass

	