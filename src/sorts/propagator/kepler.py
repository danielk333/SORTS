#!/usr/bin/env python

"""wrapper for the SGP4 propagator"""

import logging
from dataclasses import dataclass
import numpy as np
from astropy.time import TimeDelta, Time
from .base import Propagator
from sorts.types import Settings, Frames, NDArray_N, NDArray_6xN
from sorts.space_object import SpaceObject
from sorts.utils import convert_to_relative_time
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

    def propagate(
        self,
        space_object: SpaceObject,
        times: Time | TimeDelta | NDArray_N,
    ) -> NDArray_6xN:
        logger.debug("Kepler:propagate")
        tv = convert_to_relative_time(space_object.epoch, times)

        orb = space_object.state.copy()
        if space_object.frame != self.settings.internal_frame:
            orb._cart = cel.convert(
                space_object.epoch,
                orb._cart,
                in_frame=space_object.frame,
                out_frame=self.settings.internal_frame,
                frame_kwargs={},
            )
            orb.calculate_kepler()

        orb.direct_update = False
        orb.auto_update = False
        orb.solver_options = dict(
            tol=self.settings.numerical_tolerance,
            max_iter=self.settings.max_iterations,
            degree=self.settings.laguerre_degree,
        )

        orb.add(num=len(tv) - 1)
        kep0 = orb._kep[:, 0]
        orb._kep[:, :] = kep0[:, None]
        orb.propagate(tv)
        orb.calculate_cartesian()

        if self.settings.out_frame != self.settings.internal_frame:
            orb._cart = cel.convert(
                space_object.epoch + TimeDelta(tv, format="sec"),
                orb._cart,
                in_frame=self.settings.internal_frame,
                out_frame=self.settings.out_frame,
                frame_kwargs={},
            )

        logger.debug("Kepler:propagate:completed")

        return orb._cart
