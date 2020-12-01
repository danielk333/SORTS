#!/usr/bin/env python

'''Defines all the available propagators and the base-class that needs to be sub-classed to implement a custom propagator. 
Such a subclass can then be used with every other functionality of SORTS.

'''

from .base import Propagator

__all__ = [
    'Propagator',
]

try:
    from .orekit import Orekit
    __all__.append('Orekit')
except ImportError:
    Orekit = None

try:
    from .pysgp4 import SGP4
    __all__.append('SGP4')
except ImportError:
    SGP4 = None

try:
    from .rebound import Rebound
    __all__.append('Rebound')
except ImportError:
    Rebound = None

from .kepler import Kepler
__all__.append('Kepler')