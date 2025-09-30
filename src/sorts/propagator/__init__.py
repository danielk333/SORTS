#!/usr/bin/env python

"""Defines all the available propagators and the base-class that needs to be sub-classed to implement a custom propagator.
Such a subclass can then be used with every other functionality of SORTS.

"""

import importlib.util
from .base import Propagator

__all__ = [
    "Propagator",
]

# TODO: discuss with daniel if he is okay with this import guard
if importlib.util.find_spec("orekit") is not None:
    from .orekit import Orekit

    __all__.append("Orekit")
else:
    Orekit = None

try:
    from .pysgp4 import SGP4

    __all__.append("SGP4")
except ImportError:
    SGP4 = None

try:
    from .rebound import Rebound

    __all__.append("Rebound")
except ImportError:
    Rebound = None

from .kepler import Kepler

__all__.append("Kepler")

if importlib.util.find_spec("poliastro") is not None:
    from .poliastro import TwoBody

    __all__.append("TwoBody")
else:
    TwoBody = None
