#!/usr/bin/env python

"""A parent class used for interfacing any propagator."""

# Python standard import
from typing import Generic
import logging
from abc import ABC, abstractmethod

from astropy.time import Time, TimeDelta
from sorts.types import S, NDArray_N, NDArray_6xN
from sorts.utils import convert_to_relative_time
from sorts.space_object import SpaceObject

logger = logging.getLogger(__name__)


class Propagator(ABC, Generic[S]):
    def __init__(self, settings: S):
        self.settings = settings
        for key in self.settings.keys:
            logger.debug(f"Propagator:settings:{key} = {getattr(self.settings, key)}")

    def propagate_to_new_epoch(
        self,
        space_object: SpaceObject,
        dt: Time | TimeDelta | float,
        copy: bool = True,
    ) -> SpaceObject:
        """Propagate and change the epoch of this space object if the state is a `pyorb.Orbit`."""
        dt = convert_to_relative_time(space_object.epoch, dt)[0]
        new_cart = self.propagate(space_object, dt)
        if len(new_cart.shape) < 2:
            new_cart.shape = (new_cart.size, 1)
        obj = space_object.copy() if copy else space_object
        obj.state.cartesian = new_cart
        obj.state.calculate_kepler()
        obj.epoch += TimeDelta(dt, format="sec")
        return obj

    @abstractmethod
    def propagate(
        self,
        space_object: SpaceObject,
        times: Time | TimeDelta | NDArray_N,
    ) -> NDArray_6xN:
        """Propagate a state

        This function uses key-word argument to supply additional information
        to the propagator, such as area or mass.

        The coordinate frames used should be documented in the child class docstring.

        SI units are assumed unless implementation states otherwise.
        """
        pass
