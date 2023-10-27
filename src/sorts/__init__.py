#!/usr/bin/env python

"""SORTS package

"""
import ctypes
import pathlib
import sysconfig

from .version import __version__

# Find suffix
suffix = sysconfig.get_config_var("EXT_SUFFIX")
if suffix is None:
    suffix = ".so"

__sortspath__ = pathlib.Path(__file__).resolve().parent
__libpath__ = __sortspath__ / ("clibsorts" + suffix)

clibsorts = ctypes.cdll.LoadLibrary(str(__libpath__))

__all__ = []

from .propagator import __all__ as propagators

__all__ += propagators
del propagators

# classes
from .space_object import SpaceObject
from .population import Population
from .propagator import Propagator
from .radar import Scan
from .radar import Station, TX, RX
from .controller import RadarController
from .scheduler import Scheduler
from .passes import Pass
from .errors import Errors
from .simulation import Simulation
from .profiling import Profiler


# modules
from .radar import scans
from . import functions
from . import constants
from . import frames
from . import dates
from . import plotting
from . import profiling
from . import controller
from . import scheduler
from . import passes
from . import errors
from . import io
from . import interpolation
from .radar import instances as radars
from . import simulation
from . import signals
from . import correlator

# Functions
from .correlator import correlate
from .passes import equidistant_sampling
from .passes import find_passes, find_simultaneous_passes, group_passes
from .signals import hard_target_snr
from .simulation import (
    MPI_single_process,
    MPI_action,
    iterable_step,
    store_step,
    cached_step,
    iterable_cache,
    pre_post_actions,
)
from .pre_encounter import propagate_pre_encounter, distance_termination
