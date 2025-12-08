#!/usr/bin/env python

"""wrapper for the SGP4 propagator"""

import logging
from dataclasses import dataclass
import numpy as np
from astropy.time import TimeDelta, Time
from .base import Propagator
from sorts.types import Settings, Frames
from sorts.space_object import SpaceObject
import spacecoords.celestial as cel

logger = logging.getLogger(__name__)


@dataclass
class KeplerSettings(Settings):
    numerical_tolerance: float = 1e-12
    max_iterations: int = 5000
    laguerre_degree: int = 5
    out_frame: Frames = "GCRS"
    internal_frame: Frames = "GCRS"


class Kepler(Propagator[KeplerSettings]):
    """Propagator class implementing the Kepler propagator,
    the propagation always occurs in GCRS frame.

    Frame options are found in the `sorts.frames.convert` function.

    """

    def propagate(self, space_object: SpaceObject, t: TimeDelta | float, epoch: Time):
        logger.debug("Kepler:propagate")
        if isinstance(t, TimeDelta):
            tv = t.sec
        else:
            tv = t
        if not isinstance(tv, np.ndarray):
            tv = np.array([tv])

        orb = space_object.state.copy()
        if space_object.frame != self.settings.internal_frame:
            orb.cartesian = cel.convert(
                epoch,
                orb.cartesian,
                in_frame=space_object.frame,
                out_frame=self.settings.internal_frame,
                frame_kwargs={},
            )
            orb.calculate_kepler()

        orb.direct_update = False
        orb.auto_update = False
        orb.degrees = False
        orb.solver_options = dict(
            tol=self.settings.numerical_tolerance,
            max_iter=self.settings.max_iterations,
            degree=self.settings.laguerre_degree,
        )

        orb.add(num=len(tv) - 1)
        orb._kep[:, 1:] = orb._kep[:, 0][:, None]
        orb.mean_anomaly = np.mod(orb.mean_anomaly + orb.mean_motion * tv, 2 * np.pi)
        orb.calculate_cartesian()

        if self.settings.out_frame != self.settings.internal_frame:
            orb.cartesian = cel.convert(
                epoch,
                orb.cartesian,
                in_frame=self.settings.internal_frame,
                out_frame=self.settings.out_frame,
                frame_kwargs={},
            )

        logger.debug("Kepler:propagate:completed")

        return orb._cart
