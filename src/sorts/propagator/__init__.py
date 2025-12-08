#!/usr/bin/env python

"""Defines all the available propagators and the base-class that needs to be sub-classed to implement a custom propagator.
Such a subclass can then be used with every other functionality of SORTS.

"""

import importlib.util
from .base import Propagator

__all__ = [
    "Propagator",
]

from .kepler import Kepler, KeplerSettings

__all__.append("Kepler")
__all__.append("KeplerSettings")

if importlib.util.find_spec("sgp4") is not None:
    from .pysgp4 import Sgp4, Sgp4Settings
    __all__.append("Sgp4")

Rebound = None

# if importlib.util.find_spec("orekit") is not None:
#     from .orekit import Orekit
#
#     __all__.append("Orekit")
#
#
# if importlib.util.find_spec("rebound") is not None:
#     from .rebound import Rebound
#
#     __all__.append("Rebound")
#
# if importlib.util.find_spec("poliastro") is not None:
#     from .poliastro import TwoBody
#
#     __all__.append("TwoBody")
