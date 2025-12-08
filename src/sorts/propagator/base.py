#!/usr/bin/env python

"""A parent class used for interfacing any propagator."""

# Python standard import
from typing import Generic
import logging
from abc import ABC, abstractmethod

from astropy.time import Time, TimeDelta
from sorts.types import S
from sorts.space_object import SpaceObject

logger = logging.getLogger(__name__)


class Propagator(ABC, Generic[S]):
    def __init__(self, settings: S):
        self.settings = settings
        for key in self.settings.keys:
            logger.debug(f"Propagator:settings:{key} = {getattr(self.settings, key)}")

    @abstractmethod
    def propagate(self, space_object: SpaceObject, t: TimeDelta | float, epoch: Time):
        """Propagate a state

        This function uses key-word argument to supply additional information
        to the propagator, such as area or mass.

        The coordinate frames used should be documented in the child class docstring.

        SI units are assumed unless implementation states otherwise.
        """
        pass
