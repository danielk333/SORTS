#!/usr/bin/env python

'''A parent class used for interfacing any propagator.

'''

#Python standard import
from abc import ABC, abstractmethod
import inspect


#Third party import
import numpy as np
from astropy.time import Time, TimeDelta

#Local import



class Propagator(ABC):

    DEFAULT_SETTINGS = dict(
        epoch_format = 'mjd',
        epoch_scale = 'utc',
        time_format = 'sec',
        time_scale = None,
        heartbeat = False,
    )



    def __init__(self, settings=None, profiler=None, logger=None):
        self.settings = dict()
        self._check_args()
        self.profiler = profiler
        self.logger = logger

        self.settings.update(self.DEFAULT_SETTINGS)
        if settings is not None:
            self.settings.update(settings)
            self._check_settings()

        if self.logger is not None:
            for key in self.settings:
                self.logger.debug(f'Propagator:settings:{key} = {self.settings[key]}')


    def _check_settings(self):
        if self.logger is not None:
            self.logger.debug(f'Propagator:_check_settings')

        for key_s, val_s in self.settings.items():
            if key_s not in self.DEFAULT_SETTINGS:
                raise KeyError('Setting "{}" does not exist'.format(key_s))
            if type(self.DEFAULT_SETTINGS[key_s]) != type(val_s):
                raise ValueError('Setting "{}" does not support "{}"'.format(key_s, type(val_s)))

    @property
    def out_frame(self):
        if 'out_frame' in self.settings:
            return self.settings['out_frame']
        else:
            raise AttributeError('No setting called "out_frame"')


    @out_frame.setter
    def out_frame(self, val):
        if 'out_frame' in self.settings:
            self.settings['out_frame'] = val
        else:
            raise AttributeError('No setting called "out_frame"')

    @property
    def in_frame(self):
        if 'in_frame' in self.settings:
            return self.settings['in_frame']
        else:
            raise AttributeError('No setting called "in_frame"')


    @in_frame.setter
    def in_frame(self, val):
        if 'in_frame' in self.settings:
            self.settings['in_frame'] = val
        else:
            raise AttributeError('No setting called "in_frame"')


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


    def convert_time(self, t, epoch):
        '''Convert input time and epoch variables to :code:`astropy.TimeDelta` and :code:`astropy.Time` variables of the correct format and scale.
        '''
        if self.profiler is not None:
            self.profiler.start('Propagator:convert_time')
        if self.logger is not None:
            self.logger.debug(f'Propagator:convert_time')

        if epoch is None:
            pass
        elif isinstance(epoch, Time) and not isinstance(epoch, TimeDelta):
            if epoch.format != self.settings['epoch_format']:
                epoch.format = self.settings['epoch_format']

            if epoch.scale != self.settings['epoch_scale']:
                epoch = getattr(epoch,self.settings['epoch_scale'])
        else:
            epoch = Time(epoch, format=self.settings['epoch_format'], scale=self.settings['epoch_scale'])

        if len(epoch.shape) > 0:
            if epoch.size > 1:
                raise ValueError(f'Can only have one epoch, not "{epoch.size}"')
            else:
                epoch = epoch[0]


        if t is None:
            pass
        elif isinstance(t, Time) and not isinstance(t, TimeDelta):
            t = t - epoch
        elif isinstance(t, TimeDelta):
            if t.format != self.settings['time_format']:
                t.format = self.settings['time_format']

            if self.settings['time_scale'] is not None:
                if t.scale != self.settings['time_scale']:
                    t = getattr(t,self.settings['time_scale'])
        elif isinstance(t, np.ndarray):
            if np.issubdtype(t.dtype, np.datetime64):
                t = Time(t, scale=self.settings['time_scale'])
                t = t - epoch
            else:
                t = TimeDelta(t, format=self.settings['time_format'], scale=self.settings['time_scale'])
        elif isinstance(t, np.datetime64):
            t = Time(t, scale=self.settings['time_scale'])
            t = t - epoch
        else:
            t = TimeDelta(t, format=self.settings['time_format'], scale=self.settings['time_scale'])

        if self.profiler is not None:
            self.profiler.stop('Propagator:convert_time')
        if self.logger is not None:
            self.logger.debug(f'Propagator:convert_time:completed')

        return t, epoch


    def set(self, **kwargs):
        self.settings.update(kwargs)
        self._check_settings()


    @abstractmethod
    def propagate(self, t, state0, epoch, **kwargs):
        '''Propagate a state

        This function uses key-word argument to supply additional information to the propagator, such as area or mass.
        
        The coordinate frames used should be documented in the child class docstring.

        SI units are assumed unless implementation states otherwise.

        :param float/list/numpy.ndarray/astropy.TimeDelta t: Time to propagate relative the initial state epoch.
        :param float/astropy.Time epoch: The epoch of the initial state.
        :param any state0: State vector in SI-units.
        :return: State vectors in SI-units.
        '''
        return None


    def heartbeat(self, t, state, **kwargs):
        '''Function applied after propagation to time `t` and state `state`, before next time step as given in the input time vector to `propagate`.
        '''
        pass


    def __str__(self):
        return ''
