#!/usr/bin/env python

"""A parent class used for interfacing any propagator."""

# Python standard import
from typing import Generic
import logging
from abc import ABC, abstractmethod

from astropy.time import Time, TimeDelta
from sorts.types import S, NDArray_N
from sorts.space_object import SpaceObject

logger = logging.getLogger(__name__)


class Propagator(ABC, Generic[S]):
    def __init__(self, settings: S):
        self.settings = settings
        for key in self.settings.keys:
            logger.debug(f"Propagator:settings:{key} = {getattr(self.settings, key)}")

    def propagate_to_new_epoch(self, space_object, dt: TimeDelta | float):
        """Propagate and change the epoch of this space object if the state is a `pyorb.Orbit`."""
        pass
        # if "in_frame" in self.propagator.settings and "out_frame" in self.propagator.settings:
        #     out_frame = self.propagator.settings["out_frame"]
        #     self.propagator.set(out_frame=self.propagator.settings["in_frame"])
        #
        #     state = self.get_state(np.array([dt], dtype=np.float64))
        #
        #     self.propagator.set(out_frame=out_frame)
        # else:
        #     state = self.get_state(np.array([dt], dtype=np.float64))
        #
        # self.epoch = self.epoch + TimeDelta(dt, format="sec")
        #
        # x, y, z, vx, vy, vz = state.flatten()
        #
        # self.update(
        #     x=x,
        #     y=y,
        #     z=z,
        #     vx=vx,
        #     vy=vy,
        #     vz=vz,
        # )

    @abstractmethod
    def propagate(self, space_object: SpaceObject, times: Time | TimeDelta | NDArray_N):
        """Propagate a state

        This function uses key-word argument to supply additional information
        to the propagator, such as area or mass.

        The coordinate frames used should be documented in the child class docstring.

        SI units are assumed unless implementation states otherwise.
        """
        pass
