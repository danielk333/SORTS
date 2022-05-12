'''rapper for the SGP4 propagator

'''

# Python standard import
from copy import copy

# Third party import
import numpy as np
import poliastro
from poliastro.twobody.propagation import farnocchia
from poliastro.bodies import Earth
import astropy.units as units


# Local import
from .base import Propagator
from .. import frames


class TwoBody(Propagator):
    '''Propagator class implementing the Kepler propagator using `poliastro`, 
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
            method=farnocchia,
            body=Earth,
        )
    )

    def __init__(self, settings=None, **kwargs):
        super(TwoBody, self).__init__(settings=settings, **kwargs)
        if self.logger is not None:
            self.logger.debug('sorts.propagator.TwoBody:init')

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
            self.profiler.start('TwoBody:propagate')
        if self.logger is not None:
            self.logger.debug(f'TwoBody:propagate:len(t) = {len(t)}')

        t, epoch = self.convert_time(t, epoch)
        times = epoch + t

        state0_GCRS = frames.convert(
            epoch,
            state0,
            in_frame=self.settings['in_frame'],
            out_frame='GCRS',
            profiler=self.profiler,
            logger=self.logger,
        )

        poli_orb = poliastro.twobody.Orbit.from_vectors(
            self.settings['body'], 
            state0_GCRS[:3]*units.m, 
            state0_GCRS[3:]*units.m/units.s,
            epoch=epoch,
        )
        if self.profiler is not None:
            self.profiler.start('TwoBody:propagate:poliastro')
        astropy_state_GCRS = poliastro.twobody.propagation.propagate(
            poli_orb,
            t.sec*units.s, 
            method=self.settings['method'], 
            **kwargs,
        )
        states_GCRS = np.empty((6, len(t)), dtype=np.float64)
        states_GCRS[:3, :] = astropy_state_GCRS.xyz.to(units.m).value
        states_GCRS[3:, :] = astropy_state_GCRS.differentials['s'].d_xyz.to(units.m/units.s).value
        if self.profiler is not None:
            self.profiler.stop('TwoBody:propagate:poliastro')

        if self.profiler is not None:
            self.profiler.start('TwoBody:propagate:out_frame')

        states = frames.convert(
            times,
            states_GCRS,
            in_frame='GCRS',
            out_frame=self.settings['out_frame'],
            profiler=self.profiler,
            logger=self.logger,
        )

        if self.profiler is not None:
            self.profiler.stop('TwoBody:propagate:out_frame')

        if self.profiler is not None:
            self.profiler.stop('TwoBody:propagate')
        if self.logger is not None:
            self.logger.debug('TwoBody:propagate:completed')

        return states
