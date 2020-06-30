#!/usr/bin/env python

'''A parent class used for interfacing any propagator.

'''

#Python standard import
from abc import ABC, abstractmethod
import inspect


#Third party import
import numpy as np


#Local import



class Propagator(ABC):

    DEFAULT_SETTINGS = dict()

    def __init__(self, profiler=None, logger=None):
        self.settings = dict()
        self._check_args()
        self.profiler = profiler
        self.logger = logger


    def _check_args(self):
        '''This method makes sure that the function signature of the implemented are correct.
        '''
        correct_argspec = inspect.getargspec(Propagator.propagate)
        current_argspec = inspect.getargspec(self.propagate)

        correct_vars = correct_argspec.args
        current_vars = current_argspec.args

        assert len(correct_vars) == len(current_vars), 'Number of arguments in implemented get_orbit is wrong ({} not {})'.format(current_vars, correct_vars)
        for var in current_vars:
            assert var in correct_vars, 'Argument missing in implemented get_orbit, got "{}" instead'.format(var)


    def _make_numpy(self, var):
        '''Small method for converting non-numpy data structures to numpy data arrays. 
        Should be used at top of functions to minimize type checks and maximize computation speed by avoiding Python objects.
        '''
        if not isinstance(var, np.ndarray):
            if isinstance(var, float):
                var = np.array([var], dtype=np.float)
            elif isinstance(var, list):
                var = np.array(var, dtype=np.float)
            else:
                raise Exception('Input type {} not supported'.format(type(var)))
        return var


    def _check_settings(self):
        pass


    def settings(self, **kwargs):
        self.settings.update(kwargs)
        self._check_settings()


    @abstractmethod
    def propagate(self, t, state0, mjd0, **kwargs):
        '''Propagate a state

        This function uses key-word argument to supply additional information to the propagator, such as area or mass.
        
        The coordinate frames used should be documented in the child class docstring.

        SI units are assumed unless implementation states otherwise.

        :param float/list/numpy.ndarray t: Time in seconds to propagate relative the initial state epoch.
        :param float mjd0: The epoch of the initial state in fractional Julian Days.
        :param numpy.ndarray state0: 6-D Cartesian state vector in SI-units.
        :return: 6-D Cartesian state vectors in SI-units.
        '''
        return None


    def __str__(self):
        return ''
