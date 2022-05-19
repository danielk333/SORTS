#!/usr/bin/env python

'''rapper for the SGP4 propagator

'''

# Python standard import
from copy import copy

# Third party import
import numpy as np
import pyorb


# Local import
from .base import Propagator
from ...transformations import frames


class Kepler(Propagator):
    '''Propagator class implementing the Kepler propagator, 
    the propagation always occurs in GCRS frame.

    Frame options are found in the `sorts.frames.convert` function.

    :ivar str in_frame: String identifying the input frame. 
    :ivar str out_frame: String identifying the output frame. 

    :param str in_frame: String identifying the input frame.
    :param str out_frame: String identifying the output frame. 
    '''

    DEFAULT_SETTINGS = copy(Propagator.DEFAULT_SETTINGS)
    DEFAULT_SETTINGS.update(
        dict(
            out_frame='GCRS',
            in_frame='GCRS',
        )
    )

    def __init__(self, settings=None, **kwargs):
        super(Kepler, self).__init__(settings=settings, **kwargs)
        if self.logger is not None:
            self.logger.debug('sorts.propagator.Kepler:init')

    def propagate(self, t, state0, epoch, **kwargs):
        '''Propagate a state

        :param float/list/numpy.ndarray/astropy.time.TimeDelta t: Time to 
            propagate relative the initial state epoch.
        :param float/astropy.time.Time epoch: The epoch of the initial state.
        :param numpy.ndarray state0: 6-D Cartesian state vector in SI-units.
        :param bool radians: If true, all angles are assumed to be in radians.
        :return: 6-D Cartesian state vectors in SI-units.

        '''
        if self.profiler is not None:
            self.profiler.start('Kepler:propagate')
        if self.logger is not None:
            self.logger.debug(f'Kepler:propagate:len(t) = {len(t)}')

        t, epoch = self.convert_time(t, epoch)
        times = epoch + t
        tv = t.sec
        if not isinstance(tv, np.ndarray):
            tv = np.array([tv])

        if self.profiler is not None:
            self.profiler.start('Kepler:propagate:in_frame')
        if isinstance(state0, pyorb.Orbit):
            orb = state0.copy()
        elif isinstance(state0, dict):
            kw = copy(state0)
            kw.update(kwargs)
            orb = pyorb.Orbit(**kw)
            cart0 = frames.convert(
                epoch,
                orb.cartesian,
                in_frame=self.settings['in_frame'],
                out_frame='GCRS',
                profiler=self.profiler,
                logger=self.logger,
            )
            orb.cartesian = cart0
        else:
            cart0 = frames.convert(
                epoch,
                state0,
                in_frame=self.settings['in_frame'],
                out_frame='GCRS',
                profiler=self.profiler,
                logger=self.logger,
            )
            kw = {key: val for key, val in zip(
                pyorb.Orbit.CARTESIAN, cart0.flatten())}
            kw.update(kwargs)
            orb = pyorb.Orbit(**kw)
        if self.profiler is not None:
            self.profiler.stop('Kepler:propagate:in_frame')

        orb.direct_update = False
        orb.auto_update = False

        if self.profiler is not None:
            self.profiler.start('Kepler:propagate:mean_motion')

        kw_in = {
            key: val 
            for key, val in zip(pyorb.Orbit.KEPLER, orb.kepler.flatten())
        }
        orb.add(num=len(tv), **kw_in)
        orb.delete(0)
        orb.propagate(tv)
        orb.calculate_cartesian()

        if self.profiler is not None:
            self.profiler.stop('Kepler:propagate:mean_motion')
        if self.profiler is not None:
            self.profiler.start('Kepler:propagate:out_frame')

        states = frames.convert(
            times,
            orb._cart,
            in_frame='GCRS',
            out_frame=self.settings['out_frame'],
            profiler=self.profiler,
            logger=self.logger,
        )

        if self.profiler is not None:
            self.profiler.stop('Kepler:propagate:out_frame')

        if self.profiler is not None:
            self.profiler.stop('Kepler:propagate')
        if self.logger is not None:
            self.logger.debug('Kepler:propagate:completed')

        return states
